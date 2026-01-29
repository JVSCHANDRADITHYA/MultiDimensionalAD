import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = r"F:\MultiDimensionalAD\data\long_run_labelled.csv"

DROP_COLS = ["Seconds", "Timestamp_IST", "State", "label"]
SENSOR_TAGS = ("PI", "PT", "TI", "TT")

WARMUP = 50
GLOBAL_K = 2          # global PCA components
LOCAL_K = 2           # local PCA components
WINDOW = 300          # sliding window for local PCA

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

sensor_cols = [c for c in df.columns if any(tag in c for tag in SENSOR_TAGS)]
X = df[sensor_cols].values.astype(np.float64)

n_samples, n_features = X.shape
t = np.arange(n_samples)

print(f"Samples: {n_samples}, Sensors: {n_features}")

# ============================================================
# 1️⃣ OFFLINE PCA (BASELINE / GROUND TRUTH)
# ============================================================
scaler_off = StandardScaler()
X_scaled_off = scaler_off.fit_transform(X)

pca_off = PCA(n_components=2)
Z_off = pca_off.fit_transform(X_scaled_off)   # PC1, PC2 baseline

# ============================================================
# 2️⃣ GLOBAL ONLINE PCA — OJA
# ============================================================
class OjaPCA:
    def __init__(self, k=2, lr=0.002):
        self.k = k
        self.lr = lr
        self.W = None

    def partial_fit(self, x):
        if self.W is None:
            self.W = np.random.randn(self.k, x.shape[0])
            self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)

        for i in range(self.k):
            y = self.W[i] @ x
            self.W[i] += self.lr * y * (x - y * self.W[i])
            self.W[i] /= np.linalg.norm(self.W[i])
            x = x - y * self.W[i]  # deflation

    def transform(self, x):
        return np.array([w @ x for w in self.W])

# ============================================================
# 3️⃣ ONLINE PIPELINE (GLOBAL + LOCAL)
# ============================================================
scaler = StandardScaler()
scaler.partial_fit(X[:WARMUP])

oja = OjaPCA(k=GLOBAL_K, lr=0.002)
local_buffer = deque(maxlen=WINDOW)

Z_global = np.zeros((n_samples, GLOBAL_K))
Z_local = np.zeros((n_samples, LOCAL_K))

# warmup
for i in range(WARMUP):
    xs = scaler.transform(X[i:i+1])[0]
    oja.partial_fit(xs)
    local_buffer.append(xs)

# online pass
for i in range(WARMUP, n_samples):
    x = X[i:i+1]
    xs = scaler.transform(x)[0]

    # ---- inference ----
    Z_global[i] = oja.transform(xs)

    if len(local_buffer) >= LOCAL_K + 1:
        Xw = np.array(local_buffer)
        pca_local = PCA(n_components=LOCAL_K)
        Zw = pca_local.fit_transform(Xw)
        Z_local[i] = Zw[-1]
    else:
        Z_local[i] = 0.0

    # ---- update ----
    scaler.partial_fit(x)
    oja.partial_fit(xs)
    local_buffer.append(xs)

# ============================================================
# 4️⃣ COMPARISON PLOTS
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# ---------------- PC1 ----------------
axes[0].plot(t, Z_off[:, 0], color="black", linewidth=2, label="Offline PCA PC1")
axes[0].plot(t, Z_global[:, 0], label="Global Oja PC1", alpha=0.9)
axes[0].plot(t, Z_local[:, 0], label="Local PCA PC1", linestyle="--", alpha=0.6)

axes[0].set_ylabel("PC1")
axes[0].legend(loc="upper right")
axes[0].set_title("PC1 Comparison")

# ---------------- PC2 ----------------
axes[1].plot(t, Z_off[:, 1], color="black", linewidth=2, label="Offline PCA PC2")
axes[1].plot(t, Z_global[:, 1], label="Global Oja PC2", alpha=0.9)
axes[1].plot(t, Z_local[:, 1], label="Local PCA PC2", linestyle="--", alpha=0.6)

axes[1].set_ylabel("PC2")
axes[1].set_xlabel("Time")
axes[1].legend(loc="upper right")
axes[1].set_title("PC2 Comparison")

plt.suptitle(
    "Hybrid Online PCA vs Offline PCA\n"
    "Global = Oja (slow, stable) | Local = Sliding Window (fast, adaptive)"
)
plt.tight_layout()
plt.show()
