"""
Streaming LSTM Autoencoder anomaly detection with Incremental PCA (PyTorch + sklearn)
Fixed: ensures all sequence buffer vectors have the same shape to avoid np.stack errors.
"""

import os
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# ========== User parameters ==========
CSV_PATH = r"F:\MultiDimensionalAD\data\long_run_labelled.csv"
timestamp_col = None
sensor_cols = None
t_seconds = 0.00005
seq_len = 20
pca_n_components = 3
ipca_batch = 100
buffer_max = 2000
train_batch_size = 32
train_steps_per_update = 3
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Thresholding
init_warmup = 1000         # first 1000 rows assumed normal
rolling_window = 500
anomaly_sigma = 4.0
train_select_k = 1.0
SENSOR_FILTERS = ['-PT-', '-TT-', '-TI-', '-PI-', 'HSD', 'FIC']
# ====================================


def ensure_sensor_dataframe(csv_path):
    """Loads CSV if exists, else returns synthetic dataset for demo."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).dropna(axis=0, how='any')
        if SENSOR_FILTERS:
            cols = [c for c in df.columns if any(f in c for f in SENSOR_FILTERS)]
            if len(cols) == 0:
                cols = df.select_dtypes(include=np.number).columns.tolist()
        else:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        return df[cols].astype(float)
    # synthetic fallback
    print("CSV not found — generating synthetic demo data.")
    N = 5000
    t = np.arange(N)
    s1 = np.sin(0.01 * t) + 0.02 * np.random.randn(N)
    s2 = np.cos(0.015 * t) + 0.03 * np.random.randn(N)
    s3 = 0.001 * t + 0.05 * np.random.randn(N)
    # inject anomalies
    s1[1500:1510] += 3.0
    s2[3000:3010] -= 2.5
    s3[4200:4220] += 5.0
    return pd.DataFrame({"sensor1": s1, "sensor2": s2, "sensor3": s3})


# ----------------- LSTM Autoencoder -----------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=1, latent_dim=None):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim or hidden_dim
        # encoder LSTM
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, self.latent_dim)
        # decoder
        self.latent_fc = nn.Linear(self.latent_dim, hidden_dim)
        self.decoder = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]  # (B, hidden)
        latent = torch.relu(self.enc_fc(h_last))
        dec_init_h = torch.tanh(self.latent_fc(latent))
        dec_h = dec_init_h.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        dec_c = torch.zeros_like(dec_h).to(dec_h.device)
        B, T, F = x.size()
        decoder_in = torch.zeros((B, T, F)).to(x.device)
        dec_out, _ = self.decoder(decoder_in, (dec_h, dec_c))
        out = self.out_fc(dec_out)
        return out


# ----------------- Streaming trainer -----------------
def run_streaming(df):
    global sensor_cols
    if sensor_cols is None:
        sensor_cols = df.columns.tolist()
    print("Using sensor columns:", sensor_cols)

    n_sensors = len(sensor_cols)
    scaler = StandardScaler()
    # set ipca to desired number of components limited by number of sensors
    feature_dim = min(pca_n_components, n_sensors)
    ipca = IncrementalPCA(n_components=feature_dim)

    model = None
    optimizer = None
    loss_fn = nn.MSELoss()

    seq_buffer = deque(maxlen=seq_len)
    train_buffer = deque(maxlen=5000)

    history_raw = {c: deque(maxlen=buffer_max) for c in sensor_cols}
    history_recons = {c: deque(maxlen=buffer_max) for c in sensor_cols}
    history_pca = deque(maxlen=buffer_max)
    history_err = deque(maxlen=buffer_max)
    history_threshold = deque(maxlen=buffer_max)
    history_is_anom = deque(maxlen=buffer_max)

    running_errors = deque(maxlen=rolling_window)
    anomaly_indices = []
    global_idx = 0
    ipca_partial_count = 0
    rows = df[sensor_cols].values
    N = rows.shape[0]

    # --- setup live plot ---
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax1.set_title("Raw sensors (faint) + LSTM-AE reconstructions (dashed)")
    ax2.set_title("First 2 PCA components (streaming)")
    ax3.set_title("Reconstruction error and anomalies")
    plt.tight_layout()
    plt.show(block=False)

    ipca_fitted = False

    for i in range(N):
        raw = rows[i].astype(float)  # ensure float

        # update raw history
        for j, c in enumerate(sensor_cols):
            history_raw[c].append(raw[j])

        # incremental scaler update
        scaler.partial_fit(raw.reshape(1, -1))
        ipca_partial_count += 1

        # partial fit IPCA every ipca_batch samples using the last chunk
        if ipca_partial_count >= ipca_batch:
            start = max(0, i - ipca_partial_count + 1)
            chunk = rows[start:i + 1].astype(float)
            try:
                chunk_scaled = scaler.transform(chunk)
                ipca.partial_fit(chunk_scaled)
                ipca_fitted = True
            except Exception:
                pass
            ipca_partial_count = 0

        # generate a fixed-size feature vector (length = feature_dim)
        try:
            scaled = scaler.transform(raw.reshape(1, -1)).flatten()  # length = n_sensors
        except Exception:
            scaled = raw.flatten()

        if ipca_fitted:
            # IPCA transform yields length feature_dim
            pca_feat = ipca.transform(scaled.reshape(1, -1)).flatten()
        else:
            # before IPCA: take first feature_dim entries of scaled (or pad with zeros)
            if scaled.size >= feature_dim:
                pca_feat = scaled[:feature_dim].copy()
            else:
                pca_feat = np.concatenate([scaled, np.zeros(feature_dim - scaled.size, dtype=float)])

        # guarantee dtype and shape
        pca_feat = np.asarray(pca_feat, dtype=float).reshape(-1)

        # push into seq buffer (all entries have length feature_dim)
        if pca_feat.shape[0] == feature_dim:
            seq_buffer.append(pca_feat)
        else:
            # shouldn't happen, but guard
            seq_buffer.append(np.pad(pca_feat, (0, max(0, feature_dim - pca_feat.size)), 'constant'))

        # store first two components for plotting (if exist)
        if pca_feat.size >= 2:
            history_pca.append(np.array([pca_feat[0], pca_feat[1]]))
        else:
            history_pca.append(np.array([np.nan, np.nan]))

        # initialize model once we have seq_len sequences
        if model is None and len(seq_buffer) >= seq_len:
            # all entries have same shape now (feature_dim)
            model = LSTMAutoencoder(n_features=feature_dim, hidden_dim=64, num_layers=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            print("Initialized LSTM Autoencoder with input features:", feature_dim)

        recon_err = np.nan
        is_anom = False
        threshold = np.nan

        # if model ready, do reconstruction & detection
        if model is not None and len(seq_buffer) >= seq_len:
            seq = np.stack(list(seq_buffer))  # (T, feature_dim)
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, F)

            model.eval()
            with torch.no_grad():
                recon = model(seq_t).squeeze(0).cpu().numpy()  # (T, F)

            # reconstruction error for last timestep
            last_true = seq[-1]
            last_recon = recon[-1]
            err = float(np.mean((last_true - last_recon) ** 2))
            recon_err = err
            history_err.append(err)

            # compute threshold from running_errors if available
            if len(running_errors) >= 30:
                mu = float(np.mean(running_errors))
                sigma = float(np.std(running_errors)) if np.std(running_errors) > 1e-8 else 1e-8
                threshold = mu + anomaly_sigma * sigma * 1.5
                is_anom = err > threshold
            history_threshold.append(threshold if not np.isnan(threshold) else np.nan)
            history_is_anom.append(bool(is_anom))
            if is_anom:
                anomaly_indices.append(global_idx)
                print(f"[ANOMALY] idx={global_idx}, recon_err={recon_err:.6f}, threshold={threshold:.6f}")

            # inverse transform recon back to original sensor space if possible for plotting
            if ipca_fitted and recon.shape[1] == ipca.n_components:
                try:
                    recon_full = ipca.inverse_transform(recon[-1].reshape(1, -1))
                    recon_full = scaler.inverse_transform(recon_full).flatten()
                    for j, c in enumerate(sensor_cols):
                        history_recons[c].append(recon_full[j] if j < recon_full.size else np.nan)
                except Exception:
                    for c in sensor_cols:
                        history_recons[c].append(np.nan)
            else:
                for c in sensor_cols:
                    history_recons[c].append(np.nan)

            # decide whether to add this sample to train_buffer
            # decide whether to add this sample to train_buffer
            if i < init_warmup:
                # first warmup samples assumed normal
                train_buffer.append(pca_feat)
                running_errors.append(err)
            else:
                # after warmup, only add if strictly normal
                if len(running_errors) >= 30:
                    mu = float(np.mean(running_errors))
                    sigma = float(np.std(running_errors)) if np.std(running_errors) > 1e-8 else 1e-8
                    threshold = mu + anomaly_sigma * sigma
                    is_anom = err > threshold
                    if not is_anom:
                        train_buffer.append(pca_feat)
                        running_errors.append(err)
                else:
                    # fallback if not enough history
                    train_buffer.append(pca_feat)
                    running_errors.append(err)


            # online training step (periodic)
            if len(train_buffer) >= train_batch_size and (global_idx % 10 == 0):
                model.train()
                tb = np.array(train_buffer)
                if tb.shape[0] >= seq_len:
                    num_samples = min(16, tb.shape[0] - seq_len + 1)
                    starts = np.random.randint(0, tb.shape[0] - seq_len + 1, size=num_samples)
                    X = np.stack([tb[s:s + seq_len] for s in starts])
                else:
                    X = np.expand_dims(tb[-1], axis=0)  # shape (1, T, F)
                X_t = torch.tensor(X, dtype=torch.float32).to(device)
                for _ in range(train_steps_per_update):
                    optimizer.zero_grad()
                    out = model(X_t)
                    loss = loss_fn(out, X_t)
                    loss.backward()
                    optimizer.step()
                # gently pop some old samples
                for _ in range(int(len(train_buffer) * 0.02)):
                    if train_buffer:
                        train_buffer.popleft()
        else:
            # model not ready yet: placeholders
            history_err.append(np.nan)
            history_threshold.append(np.nan)
            history_is_anom.append(False)
            for c in sensor_cols:
                history_recons[c].append(np.nan)

        # live plotting (update every samples)
        #stop plotting only save anomaly values, comment all plotting code
        if i % 1 == 0 or i == N - 1:
            xs = np.arange(len(history_err))
            ys = np.array(history_err, dtype=float)
            thr = np.array(history_threshold, dtype=float)
            is_anom_arr = np.array(history_is_anom, dtype=bool)

            ax1.clear()
            for c in sensor_cols:
                ax1.plot(xs, list(history_raw[c]), alpha=0.12, label=f"{c} raw")
                ax1.plot(xs, list(history_recons[c]), linestyle="--", label=f"{c} recon")
            ax1.legend(fontsize="small", loc="upper right")
            ax1.set_title("Raw sensors (faint) + LSTM-AE reconstructions (dashed)")

            ax2.clear()
            if len(history_pca) > 0:
                comp1 = [v[0] if not np.isnan(v[0]) else np.nan for v in history_pca]
                comp2 = [v[1] if not np.isnan(v[1]) else np.nan for v in history_pca]
                ax2.plot(xs, comp1, label="PCA comp1")
                ax2.plot(xs, comp2, label="PCA comp2")
                ax2.legend(loc="upper right")
            ax2.set_title("First 2 PCA components (streaming)")

            ax3.clear()
            ax3.plot(xs, ys, label="reconstruction error")
            if not np.all(np.isnan(thr)):
                ax3.plot(xs, thr, label="threshold (mean+k*std)")
            if is_anom_arr.any():
                ax3.scatter(xs[is_anom_arr], ys[is_anom_arr], color='red', s=12, label="anomaly")
            ax3.legend(loc="upper right")
            ax3.set_title("Reconstruction error and anomalies")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        global_idx += 1

    plt.ioff()
    plt.show()
    return {
        "sensor_history": {c: np.array(history_raw[c]) for c in sensor_cols},
        "recon_history": {c: np.array(history_recons[c]) for c in sensor_cols},
        "pca_history": np.array(history_pca),
        "recon_err": np.array(history_err),
        "threshold": np.array(history_threshold),
        "anomalies": anomaly_indices
    }


if __name__ == "__main__":
    df = ensure_sensor_dataframe(CSV_PATH)
    results = run_streaming(df)
    print("Detected anomalies indices (stream index):", results["anomalies"])
    
    # save all anomaies with indexes and timestamps if available
    if timestamp_col and timestamp_col in df.columns:
        anomaly_times = df.iloc[results["anomalies"]][timestamp_col].values
        anomaly_list = list(zip(results["anomalies"], anomaly_times))
        print("Anomalies with timestamps:")
        for idx, ts in anomaly_list:
            print(f"Index: {idx}, Timestamp: {ts}")

