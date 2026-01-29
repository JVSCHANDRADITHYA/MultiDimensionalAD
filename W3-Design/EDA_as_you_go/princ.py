import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
CSV_PATH = r"F:\MultiDimensionalAD\data\long_run_labelled.csv"

DROP_COLS = ["Seconds", "Timestamp_IST", "State", "label"]
SENSOR_TAGS = ("PI", "PT", "TI", "TT")

LATENT_DIM = 32
TRAIN_EPOCHS = 25
BATCH_SIZE = 128

STREAM_HZ = 2000
MAX_POINTS = 400
THRESH_SIGMA = 4.0

# pick sensors to visualize
PLOT_SENSORS = [0, 1, 2, 3]   # indices after filtering

# ================= LOAD =================
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
sensor_cols = [c for c in df.columns if any(tag in c for tag in SENSOR_TAGS)]
X = df[sensor_cols].values.astype(np.float32)

n_samples, n_sensors = X.shape
print(f"Sensors: {n_sensors}, Samples: {n_samples}")

# ================= SCALE =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= AUTOENCODER =================
class SensorAE(nn.Module):
    def __init__(self, n_sensors, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_sensors, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_sensors)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = SensorAE(n_sensors, LATENT_DIM)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ================= TRAIN =================
dataset = torch.tensor(X_scaled)
loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

model.train()
for e in range(TRAIN_EPOCHS):
    lsum = 0
    for b in loader:
        opt.zero_grad()
        loss = loss_fn(model(b), b)
        loss.backward()
        opt.step()
        lsum += loss.item()
    print(f"Epoch {e+1}: {lsum/len(loader):.6f}")

# ================= OFFLINE RECON =================
model.eval()
with torch.no_grad():
    X_recon = model(dataset).numpy()

# per-sample error for threshold
err = np.mean((X_scaled - X_recon) ** 2, axis=1)
mu, sigma = err.mean(), err.std()
THRESH = mu + THRESH_SIGMA * sigma
print(f"Anomaly threshold: {THRESH:.4f}")

# ================= FULL STATIC PLOT =================
fig, axes = plt.subplots(len(PLOT_SENSORS), 1, figsize=(16, 8), sharex=True)

for i, s in enumerate(PLOT_SENSORS):
    axes[i].plot(X_scaled[:, s], label="Original", linewidth=1)
    axes[i].plot(X_recon[:, s], label="Reconstructed", linewidth=1, alpha=0.7)
    axes[i].set_ylabel(sensor_cols[s])
    axes[i].legend()

axes[-1].set_xlabel("Time")
fig.suptitle("Offline Reconstruction: Sensor vs Reconstructed")
plt.tight_layout()
plt.show()

# ================= LIVE PLOT =================
plt.ion()
fig, axes = plt.subplots(len(PLOT_SENSORS), 1, figsize=(16, 8), sharex=True)

lines_orig = []
lines_rec = []
scatters = []

for i, s in enumerate(PLOT_SENSORS):
    lo, = axes[i].plot([], [], label="Original")
    lr, = axes[i].plot([], [], label="Recon", alpha=0.7)
    sc = axes[i].scatter([], [], color="red", s=30)
    axes[i].set_ylabel(sensor_cols[s])
    axes[i].legend()

axes[-1].set_xlabel("Time")

t_hist = []
orig_hist = [[] for _ in PLOT_SENSORS]
rec_hist = [[] for _ in PLOT_SENSORS]
anom_x = [[] for _ in PLOT_SENSORS]
anom_y = [[] for _ in PLOT_SENSORS]

# ================= STREAM =================
for t in range(n_samples):
    x = X[t:t+1]
    xs = scaler.transform(x)

    with torch.no_grad():
        xr = model(torch.tensor(xs)).numpy()

    e = np.mean((xs - xr) ** 2)

    t_hist.append(t)

    for i, s in enumerate(PLOT_SENSORS):
        orig_hist[i].append(xs[0, s])
        rec_hist[i].append(xr[0, s])

        if e > THRESH:
            anom_x[i].append(t)
            anom_y[i].append(xs[0, s])

        # trim
        if len(t_hist) > MAX_POINTS:
            t_hist = t_hist[-MAX_POINTS:]
            orig_hist[i] = orig_hist[i][-MAX_POINTS:]
            rec_hist[i] = rec_hist[i][-MAX_POINTS:]
            anom_x[i] = anom_x[i][-MAX_POINTS:]
            anom_y[i] = anom_y[i][-MAX_POINTS:]

        axes[i].lines[0].set_data(t_hist, orig_hist[i])
        axes[i].lines[1].set_data(t_hist, rec_hist[i])
        axes[i].collections[0].set_offsets(np.c_[anom_x[i], anom_y[i]])

        axes[i].relim()
        axes[i].autoscale_view()

    plt.pause(0.001)
    time.sleep(1.0 / STREAM_HZ)

plt.ioff()
plt.show()
