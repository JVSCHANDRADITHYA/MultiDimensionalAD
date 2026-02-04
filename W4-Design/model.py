import pandas as pd
import numpy as np
from collections import deque, defaultdict
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
DATA_PATH = r"F:\MultiDimensionalAD\data\real_time_MKPL\202505271525GMT+5x30.h24.csv"

PRESSURE_TAGS = ("PI",)
OP_TAGS = ("MOV", "PUMP", "MP", "SBV", "SCR", "XXI")

CTX = 10
COLD_START = 200
RESID_WIN = 50
USE_PCA = True

# mode-aware thresholds
MODE_THRESH = {
    "STEADY":     {"z": 4.5},
    "TRANSIENT": {"z": 8.0},
    "SCRAPER":   {"z": np.inf}
}

# =============================
# LOAD & CLEAN
# =============================
df = pd.read_csv(DATA_PATH)

# drop timestamp explicitly (THIS FIXES YOUR CRASH)
for col in ["Timestamp_IST", "Seconds"]:
    if col in df.columns:
        df = df.drop(columns=[col])

pressure_cols = [c for c in df.columns if any(k in c for k in PRESSURE_TAGS)]
op_cols = [c for c in df.columns if any(k in c for k in OP_TAGS)]

print(f"[INFO] Pressure sensors: {len(pressure_cols)}")
print(f"[INFO] Operational tags: {len(op_cols)}")

# =============================
# HELPERS
# =============================
def infer_mode(row):
    if row.filter(like="SCR").sum() > 0:
        return "SCRAPER"
    if row.filter(like="MOV").diff().abs().sum() > 0:
        return "TRANSIENT"
    return "STEADY"

# =============================
# STATE
# =============================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

buffers = {c: deque(maxlen=CTX) for c in pressure_cols}
residuals = {c: deque(maxlen=RESID_WIN) for c in pressure_cols}

models = {
    c: SGDRegressor(
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        warm_start=True
    )
    for c in pressure_cols
}

model_ready = {c: False for c in pressure_cols}

events = []          # (t, sensor, z, mode)
final_label = defaultdict(lambda: "OK")

# =============================
# STREAMING LOOP
# =============================
for t, row in df.iterrows():

    mode = infer_mode(row)

    ops = pd.to_numeric(row[op_cols], errors="coerce").fillna(0.0).values
    pres = pd.to_numeric(row[pressure_cols], errors="coerce").values
    if np.isnan(pres).any():
        continue

    # update scalers during cold start
    if t < COLD_START:
        x_scaler.partial_fit([ops])
        y_scaler.partial_fit(pres.reshape(-1, 1))
        for i, c in enumerate(pressure_cols):
            buffers[c].append(pres[i])
        continue

    ops_scaled = x_scaler.transform([ops])[0]

    for i, c in enumerate(pressure_cols):
        buffers[c].append(pres[i])

        if len(buffers[c]) < CTX:
            continue

        hist = np.array(buffers[c])
        feat = np.hstack([ops_scaled, hist[:-1]])

        y = pres[i]
        y_s = y_scaler.transform([[y]])[0, 0]

        # bootstrap
        if not model_ready[c]:
            models[c].partial_fit([feat], [y_s])
            model_ready[c] = True
            continue

        y_pred = models[c].predict([feat])[0]
        resid = y_s - y_pred
        residuals[c].append(resid)

        if len(residuals[c]) >= 20:
            mu = np.mean(residuals[c])
            sig = np.std(residuals[c]) + 1e-6
            z = abs(resid - mu) / sig

            if z > MODE_THRESH[mode]["z"]:
                events.append((t, c, z, mode))
                final_label[c] = "ANOMALOUS"

        models[c].partial_fit([feat], [y_s])

# =============================
# SUMMARY TABLE
# =============================
summary = pd.DataFrame({
    "sensor": pressure_cols,
    "final_label": [final_label[c] for c in pressure_cols],
    "anomaly_count": [
        sum(1 for e in events if e[1] == c) for c in pressure_cols
    ]
})

print("\n=== SENSOR SUMMARY ===")
print(summary.sort_values("anomaly_count", ascending=False).head(20))

# =============================
# ANOMALY PLOT
# =============================
plt.figure(figsize=(14, 5))
for c in pressure_cols:
    ts = [e[0] for e in events if e[1] == c]
    if ts:
        plt.scatter(ts, [c]*len(ts), s=8)

plt.title("Anomaly Points per Sensor (Residual-based)")
plt.xlabel("Time index")
plt.ylabel("Sensor")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# OPTIONAL PCA ON RESIDUALS
# =============================
if USE_PCA:
    res_mat = []
    for c in pressure_cols:
        if len(residuals[c]) == RESID_WIN:
            res_mat.append(residuals[c])

    if len(res_mat) >= 3:
        res_mat = np.array(res_mat).T
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(res_mat)

        plt.figure(figsize=(10,4))
        plt.plot(pcs[:,0], label="PC1 residual")
        plt.plot(pcs[:,1], label="PC2 residual")
        plt.title("PCA on Residuals (Context-aware)")
        plt.legend()
        plt.grid(True)
        plt.show()

print("\n[INFO] Done.")
