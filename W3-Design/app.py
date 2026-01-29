import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# ---------------- CONFIG ----------------
CSV_PATH = r"F:\MultiDimensionalAD\data\long_run_labelled.csv"
STREAM_HZ = 200
WARMUP = 50
IPCA_BATCH = 20
MAX_POINTS = 1000

DROP_COLS = ["Seconds", "Timestamp_IST", "State", "label"]
SENSOR_TAGS = ("PI", "PT", "TI", "TT")

# ---------------- LOAD ----------------
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# select only pressure / temperature sensors
sensor_cols = [
    c for c in df.columns
    if any(tag in c for tag in SENSOR_TAGS)
]

df_sensors = df[sensor_cols]

X = df_sensors.values.astype(np.float64)
n_samples, n_features = X.shape

print(f"Using {n_features} sensor channels for PCA")

# ---------------- MODELS ----------------
scaler = StandardScaler()
ipca = IncrementalPCA(n_components=3)

# ---------------- WARMUP ----------------
scaler.fit(X[:WARMUP])
ipca.fit(scaler.transform(X[:WARMUP]))

# ---------------- PLOT SETUP ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(15, 5))

time_hist = []
pc1_hist, pc2_hist, pc3_hist = [], [], []
sensor_hist = [[] for _ in range(n_features)]
buffer = []

# PCA lines
pc_lines = [
    ax.plot([], [], label="PC1", linewidth=2, alpha=0.7)[0],
    ax.plot([], [], label="PC2", linewidth=2, alpha=0.6)[0],
    ax.plot([], [], label="PC3", linewidth=2, alpha=0.5)[0],
]

# sensor lines (faint)
sensor_lines = [
    ax.plot([], [], alpha=0.25, linewidth=0.8, color="gray")[0]
    for _ in range(n_features)
]

ax.legend(loc="upper right")
ax.set_title("Live PCA (Pressure & Temperature Sensors Only)")
ax.set_xlabel("Time index")
ax.set_ylabel("Scaled value")

# ---------------- STREAM ----------------
for t in range(WARMUP, n_samples):

    x = X[t].reshape(1, -1)

    # fast path
    xs = scaler.transform(x)
    z = ipca.transform(xs)[0]  # PC1, PC2, PC3

    time_hist.append(t)
    pc1_hist.append(z[0])
    pc2_hist.append(z[1])
    pc3_hist.append(z[2])

    for i in range(n_features):
        sensor_hist[i].append(xs[0, i])

    # trim history
    if len(time_hist) > MAX_POINTS:
        time_hist = time_hist[-MAX_POINTS:]
        pc1_hist = pc1_hist[-MAX_POINTS:]
        pc2_hist = pc2_hist[-MAX_POINTS:]
        pc3_hist = pc3_hist[-MAX_POINTS:]
        for i in range(n_features):
            sensor_hist[i] = sensor_hist[i][-MAX_POINTS:]

    # update PCA lines
    pc_lines[0].set_data(time_hist, pc1_hist)
    pc_lines[1].set_data(time_hist, pc2_hist)
    pc_lines[2].set_data(time_hist, pc3_hist)

    # update sensor lines
    for i in range(n_features):
        sensor_lines[i].set_data(time_hist, sensor_hist[i])

    ax.relim()
    ax.autoscale_view()

    plt.pause(0.001)

    # slow adaptation
    buffer.append(x[0])
    if len(buffer) >= IPCA_BATCH:
        buf = np.array(buffer)
        scaler.partial_fit(buf)
        ipca.partial_fit(scaler.transform(buf))
        buffer.clear()

    time.sleep(1.0 / STREAM_HZ)
