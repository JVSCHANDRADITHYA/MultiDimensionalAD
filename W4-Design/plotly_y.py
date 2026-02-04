import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================
# CONFIG
# =============================
DATA_PATH = r"F:\MultiDimensionalAD\data\real_time_MKPL\202505271525GMT+5x30.h24.csv"

PRESSURE_TAGS = ("PI",)
N_COMPONENTS = 2
CTX_LEN = 50
CHUNK_SIZE = 1
MAX_POINTS = 500
PAUSE_SEC = 0.01

# =============================
# LOAD FULL DATA (offline PCA)
# =============================
df = pd.read_csv(DATA_PATH)

pressure_cols = [c for c in df.columns if any(tag in c for tag in PRESSURE_TAGS)]
if not pressure_cols:
    raise ValueError("No pressure sensors found")

print(f"[INFO] Using {len(pressure_cols)} pressure sensors")

X_full = df[pressure_cols].values

offline_scaler = StandardScaler()
X_full_scaled = offline_scaler.fit_transform(X_full)

offline_pca = PCA(n_components=N_COMPONENTS)
X_pca_offline = offline_pca.fit_transform(X_full_scaled)

pc1_offline = X_pca_offline[:, 0]
pc2_offline = X_pca_offline[:, 1]

# =============================
# ONLINE STATE
# =============================
online_scaler = StandardScaler()

pca_window = []
sensor_buffer = []
pc1_online = []
pc2_online = []
time_buffer = []

t_global = 0

# =============================
# PLOT SETUP
# =============================
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# ---- TOP: pressure sensors (subset, readable) ----
plot_cols = np.random.choice(
    pressure_cols, size=min(10, len(pressure_cols)), replace=False
)

pressure_lines = []
for _ in plot_cols:
    line, = ax1.plot([], [], linewidth=0.9)
    pressure_lines.append(line)

ax1.set_title("Pressure Sensors (Streaming, Scaled)")
ax1.set_ylabel("Scaled pressure")
ax1.grid(True)
ax1.legend(plot_cols, loc="upper right", fontsize=8)

# ---- BOTTOM: PCA + background sensors ----
# background pressure lines (ALL sensors)
bg_lines = []
for _ in pressure_cols:
    line, = ax2.plot([], [], color="lightgray", alpha=0.3, linewidth=0.8, zorder=1)
    bg_lines.append(line)

# PCA lines
pc1_on_line, = ax2.plot([], [], label="PC1 sliding", linewidth=2, zorder=10)
pc2_on_line, = ax2.plot([], [], label="PC2 sliding", linewidth=2, zorder=10)

pc1_off_line, = ax2.plot([], [], "--", color="gray", alpha=0.6, label="PC1 offline", zorder=9)
pc2_off_line, = ax2.plot([], [], "--", color="black", alpha=0.6, label="PC2 offline", zorder=9)

ax2.set_title("Sliding Window PCA with Background Pressure Sensors")
ax2.set_xlabel("Time index")
ax2.set_ylabel("Component / Scaled value")
ax2.legend()
ax2.grid(True)

# =============================
# STREAMING LOOP
# =============================
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    x = chunk[pressure_cols].values  # (1, n_pressure)

    # ---- scaling ----
    online_scaler.partial_fit(x)
    x_scaled = online_scaler.transform(x)[0]

    # ---- update PCA window ----
    pca_window.append(x_scaled)
    if len(pca_window) > CTX_LEN:
        pca_window.pop(0)

    # ---- PCA compute ----
    if len(pca_window) == CTX_LEN:
        pca = PCA(n_components=N_COMPONENTS)
        pcs = pca.fit_transform(np.array(pca_window))
        pc_latest = pcs[-1]

        pc1_online.append(pc_latest[0])
        pc2_online.append(pc_latest[1])
        time_buffer.append(t_global)

    # ---- store sensor buffer ----
    sensor_buffer.append(x_scaled)
    if len(sensor_buffer) > MAX_POINTS:
        sensor_buffer.pop(0)

    if len(time_buffer) > MAX_POINTS:
        pc1_online.pop(0)
        pc2_online.pop(0)
        time_buffer.pop(0)

    sensor_arr = np.array(sensor_buffer)

    # ---- update TOP plot (subset) ----
    for i, line in enumerate(pressure_lines):
        idx = pressure_cols.index(plot_cols[i])
        line.set_data(
            range(t_global - len(sensor_arr) + 1, t_global + 1),
            sensor_arr[:, idx]
        )

    ax1.relim()
    ax1.autoscale_view()

    # ---- update BOTTOM plot background sensors ----
    for i, line in enumerate(bg_lines):
        if i < sensor_arr.shape[1]:
            line.set_data(
                range(t_global - len(sensor_arr) + 1, t_global + 1),
                sensor_arr[:, i]
            )

    # ---- update PCA curves ----
    if time_buffer:
        pc1_on_line.set_data(time_buffer, pc1_online)
        pc2_on_line.set_data(time_buffer, pc2_online)

        pc1_off_line.set_data(time_buffer, pc1_offline[time_buffer])
        pc2_off_line.set_data(time_buffer, pc2_offline[time_buffer])

        ax2.relim()
        ax2.autoscale_view()

    plt.pause(PAUSE_SEC)
    t_global += 1

plt.ioff()
plt.show()
