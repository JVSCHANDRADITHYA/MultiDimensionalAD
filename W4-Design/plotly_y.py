import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA

# =============================
# CONFIG
# =============================
DATA_PATH = r"F:\MultiDimensionalAD\data\202505161524GMT+5x30.h24.csv"

PRESSURE_TAGS = ("PI",)        # ONLY pressure sensors
N_COMPONENTS = 2
CHUNK_SIZE = 1
MAX_POINTS = 500
PAUSE_SEC = 0.01

# =============================
# LOAD FULL DATA ONCE (for offline PCA)
# =============================
df = pd.read_csv(DATA_PATH)

pressure_cols = [c for c in df.columns if any(tag in c for tag in PRESSURE_TAGS)]
if not pressure_cols:
    raise ValueError("No pressure sensors found")

print(f"[INFO] Using {len(pressure_cols)} pressure sensors")

X_full = df[pressure_cols].values

# =============================
# OFFLINE PCA (REFERENCE)
# =============================
offline_scaler = StandardScaler()
X_full_scaled = offline_scaler.fit_transform(X_full)

offline_pca = PCA(n_components=N_COMPONENTS)
X_pca_offline = offline_pca.fit_transform(X_full_scaled)

pc1_offline = X_pca_offline[:, 0]
pc2_offline = X_pca_offline[:, 1]

# =============================
# ONLINE MODELS
# =============================
online_scaler = StandardScaler()
ipca = IncrementalPCA(n_components=N_COMPONENTS)

# =============================
# BUFFERS
# =============================
sensor_buffer = []
pc1_online = []
pc2_online = []
time_buffer = []

warmup_scaled = []
pca_ready = False
t_global = 0

# =============================
# PLOT SETUP
# =============================
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# ---- Pressure sensors ----
pressure_lines = []
for col in np.random.choice(pressure_cols, size=min(5, len(pressure_cols)), replace=False):
    line, = ax1.plot([], [], linewidth=0.8)
    pressure_lines.append(line)

ax1.set_title("Pressure Sensors (Streaming)")
ax1.set_ylabel("Pressure value")
ax1.grid(True)

# ---- PCA lines ----
pc1_on_line, = ax2.plot([], [], label="PC1 online", linewidth=2)
pc2_on_line, = ax2.plot([], [], label="PC2 online", linewidth=2)

pc1_off_line, = ax2.plot(
    [], [], "--", color="gray", alpha=0.5, label="PC1 offline"
)
pc2_off_line, = ax2.plot(
    [], [], "--", color="black", alpha=0.5, label="PC2 offline"
)

ax2.set_title("PCA vs Time (Online vs Offline)")
ax2.set_xlabel("Time index")
ax2.set_ylabel("Component value")
ax2.legend()
ax2.grid(True)

# =============================
# STREAMING LOOP
# =============================
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    x = chunk[pressure_cols].values  # (1, n_pressure)

    # ---- scaling ----
    online_scaler.partial_fit(x)
    x_scaled = online_scaler.transform(x)

    # ---- PCA warm-up ----
    if not pca_ready:
        warmup_scaled.append(x_scaled[0])

        if len(warmup_scaled) < N_COMPONENTS:
            t_global += 1
            continue

        warmup_scaled = np.array(warmup_scaled)
        ipca.partial_fit(warmup_scaled)
        pcs = ipca.transform(warmup_scaled)

        for i in range(N_COMPONENTS):
            sensor_buffer.append(warmup_scaled[i])
            pc1_online.append(pcs[i, 0])
            pc2_online.append(pcs[i, 1])
            time_buffer.append(t_global - N_COMPONENTS + 1 + i)

        pca_ready = True
        continue

    # ---- online step ----
    ipca.partial_fit(x_scaled)
    pc = ipca.transform(x_scaled)[0]

    sensor_buffer.append(x.flatten())
    pc1_online.append(pc[0])
    pc2_online.append(pc[1])
    time_buffer.append(t_global)

    # ---- rolling window ----
    if len(sensor_buffer) > MAX_POINTS:
        sensor_buffer.pop(0)
        pc1_online.pop(0)
        pc2_online.pop(0)
        time_buffer.pop(0)

    sensor_arr = np.array(sensor_buffer)

    # ---- update pressure plot ----
    for i, line in enumerate(pressure_lines):
        line.set_data(time_buffer, sensor_arr[:, i])

    ax1.relim()
    ax1.autoscale_view()

    # ---- update PCA plot ----
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
