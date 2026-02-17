import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ================= CONFIG =================

WINDOW = 50
COLD_START = 100
PCA_COMPONENTS = 1
AE_SEQ_LEN = 20
PEER_Z = 3.0
REG_MIN = 30

DATA_PATH = r"F:\MultiDimensionalAD\data\202505161524GMT+5x30.h24.csv"

# ================= SENSOR FILTERS =================

MAIN_SENSORS = {
    "P": ["PI", "PIC"],
    "T": ["TI", "TIC"],
    "F": ["FI", "FIC"]
}

OP_KEYS = ["MOV", "PUMP", "SCR", "XXI", "DRA", "SBV", "TYPE", "MP"]

SENSOR_STATES = [
    "HEALTHY",
    "NON-OPERATIONAL",
    "DEVIATING",
    "OPERATIONAL_DRIVEN",
    "EXCLUDED"
]

# ================= HELPERS =================

def classify(col):
    for k in OP_KEYS:
        if k in col:
            return "OP"
    for g, keys in MAIN_SENSORS.items():
        if any(k in col for k in keys):
            return g
    return None

# ================= LSTM AE =================

class LSTMAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.LSTM(dim, 16, batch_first=True)
        self.dec = nn.LSTM(16, dim, batch_first=True)

    def forward(self, x):
        z, _ = self.enc(x)
        out, _ = self.dec(z)
        return out

# ================= LOAD DATA =================

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Timestamp_IST", "Seconds"], errors="ignore")

groups = defaultdict(list)
operational = []

for c in df.columns:
    t = classify(c)
    if t == "OP":
        operational.append(c)
    elif t:
        groups[t].append(c)

# ================= STATE =================

buffers = defaultdict(lambda: deque(maxlen=WINDOW))
state = {c: "HEALTHY" for c in df.columns}

# PCA
pca_models = {g: IncrementalPCA(n_components=1) for g in ["P", "T", "F"]}
scalers = {g: StandardScaler() for g in ["P", "T", "F"]}
pca_buffers = {g: deque(maxlen=WINDOW) for g in ["P", "T", "F"]}
PCA_SENSORS = {g: None for g in ["P", "T", "F"]}

# AE
latent_seq = deque(maxlen=AE_SEQ_LEN)
ae = LSTMAE(3)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Regression
recon_err_buf = deque(maxlen=WINDOW)
op_energy_buf = deque(maxlen=WINDOW)
reg = LinearRegression()

sensor_log = []
system_log = []

# ================= LIVE PLOT SETUP =================

fig, ax = plt.subplot_mosaic(
    [
        ["A", "B", "C"],
        ["D", "E", "E"],
    ],
    constrained_layout=True,
    figsize=(15, 9)
)

ax_pressure = ax["A"]
ax_temp     = ax["B"]
ax_flow     = ax["C"]

ax_avg = ax["D"]     # avg P/T/F
ax_pca = ax["E"]     # PC1 per group

ax_pressure.set_title("Pressure Sensors")
ax_temp.set_title("Temperature Sensors")
ax_flow.set_title("Flow Sensors")

ax_avg.set_title("Average P / T / F")
ax_pca.set_title("PCA (PC1) per Group")

for a in [ax_pressure, ax_temp, ax_flow, ax_avg, ax_pca]:
    a.set_xlabel("Time")
    a.set_ylabel("Value")
    a.grid(True)

# ---- line holders ----
pressure_lines = {}
temp_lines = {}
flow_lines = {}

avg_p_line, = ax_avg.plot([], [], label="Avg P")
avg_t_line, = ax_avg.plot([], [], label="Avg T")
avg_f_line, = ax_avg.plot([], [], label="Avg F")
ax_avg.legend()

pc_p_line, = ax_pca.plot([], [], label="P_PC1")
pc_t_line, = ax_pca.plot([], [], label="T_PC1")
pc_f_line, = ax_pca.plot([], [], label="F_PC1")
ax_pca.legend()


# ================= MAIN LOOP =================

for t, row in df.iterrows():

    # ---- update buffers ----
    for c, v in row.items():
        if np.isfinite(v):
            buffers[c].append(v)

    # ---- cold start ----
    for c, buf in buffers.items():
        if len(buf) == COLD_START:
            if np.std(buf) < 1e-6:
                state[c] = "NON-OPERATIONAL"
                state[c] = "EXCLUDED"

    # ---- peer deviation ----
    for g, cols in groups.items():
        healthy = [c for c in cols if state[c] == "HEALTHY"]
        if len(healthy) < 3:
            continue

        vals = np.array([buffers[c][-1] for c in healthy])
        med, std = np.median(vals), np.std(vals) + 1e-6

        for c, v in zip(healthy, vals):
            if abs(v - med) / std > PEER_Z:
                state[c] = "DEVIATING"

    # ---- freeze PCA sensors ----
    for g in ["P", "T", "F"]:
        if PCA_SENSORS[g] is None:
            healthy = [c for c in groups[g] if state[c] == "HEALTHY"]
            if len(healthy) >= 2:
                PCA_SENSORS[g] = healthy.copy()

    pcs = []

    # ---- PCA per group ----
    for g in ["P", "T", "F"]:
        if PCA_SENSORS[g] is None:
            pcs.append(0.0)
            continue

        x = np.array([[buffers[c][-1] for c in PCA_SENSORS[g]]])

        if len(pca_buffers[g]) < WINDOW:
            pca_buffers[g].append(x.flatten())
            pcs.append(0.0)
            continue

        X = scalers[g].fit_transform(np.vstack(pca_buffers[g]))
        pca_models[g].partial_fit(X)

        pc = pca_models[g].transform(scalers[g].transform(x))[0, 0]
        pcs.append(pc)

        pca_buffers[g].append(x.flatten())

    latent_seq.append(pcs)

    # ---- operational energy ----
    op_vals = [abs(row[c]) for c in operational if np.isfinite(row[c])]
    op_energy = np.mean(op_vals) if op_vals else 0.0
    op_energy_buf.append(op_energy)

    # ---- AE ----
    if len(latent_seq) == AE_SEQ_LEN:
        x = torch.from_numpy(np.stack(latent_seq)).float().unsqueeze(0)
        xr = ae(x)
        loss = loss_fn(xr, x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        recon_err = loss.item()
        recon_err_buf.append(recon_err)
    else:
        recon_err = 0.0

    # ---- threshold regression ----
    if len(op_energy_buf) >= REG_MIN and len(recon_err_buf) >= REG_MIN:
        X = np.array(list(op_energy_buf)[-REG_MIN:]).reshape(-1, 1)
        y = np.array(list(recon_err_buf)[-REG_MIN:])
        reg.fit(X, y)
        threshold = reg.predict([[op_energy]])[0]
    else:
        threshold = np.mean(recon_err_buf) if recon_err_buf else 0.0

    # ---- decision ----
    if recon_err > threshold:
        decision = "OPERATIONAL_DRIVEN" if op_energy > np.mean(op_energy_buf) else "LEAK"
    else:
        decision = "NORMAL"

    # ================= LIVE PLOTTING =================

    # ---- Pressure / Temp / Flow ----
    for s in groups["P"]:
        if s not in pressure_lines:
            pressure_lines[s], = ax_pressure.plot([], [], lw=0.8)
        y = list(buffers[s])
        x = np.arange(len(y))
        pressure_lines[s].set_data(x, y)

    for s in groups["T"]:
        if s not in temp_lines:
            temp_lines[s], = ax_temp.plot([], [], lw=0.8)
        y = list(buffers[s])
        x = np.arange(len(y))
        temp_lines[s].set_data(x, y)

    for s in groups["F"]:
        if s not in flow_lines:
            flow_lines[s], = ax_flow.plot([], [], lw=0.8)
        y = list(buffers[s])
        x = np.arange(len(y))
        flow_lines[s].set_data(x, y)

    for a in [ax_pressure, ax_temp, ax_flow]:
        a.relim()
        a.autoscale_view()

    # ---- Averages ----
    if len(recon_err_buf) > 0:
        avg_p = np.mean([buffers[c][-1] for c in groups["P"] if buffers[c]])
        avg_t = np.mean([buffers[c][-1] for c in groups["T"] if buffers[c]])
        avg_f = np.mean([buffers[c][-1] for c in groups["F"] if buffers[c]])

        avg_p_line.set_data(range(len(recon_err_buf)), list(recon_err_buf))
        avg_t_line.set_data(range(len(recon_err_buf)), list(recon_err_buf))
        avg_f_line.set_data(range(len(recon_err_buf)), list(recon_err_buf))

        ax_avg.relim()
        ax_avg.autoscale_view()

    # ---- PCA PC1s ----
    pc_hist = np.array(system_log[-WINDOW:]) if len(system_log) > 0 else None
    if pc_hist is not None:
        pc_p_line.set_data(range(len(pc_hist)), [x["P_PC1"] for x in pc_hist])
        pc_t_line.set_data(range(len(pc_hist)), [x["T_PC1"] for x in pc_hist])
        pc_f_line.set_data(range(len(pc_hist)), [x["F_PC1"] for x in pc_hist])

        ax_pca.relim()
        ax_pca.autoscale_view()

    plt.pause(0.001)

    
    # ---- logs ----
    for c in df.columns:
        sensor_log.append({
            "t": t,
            "sensor": c,
            "state": state[c]
        })

    system_log.append({
        "t": t,
        "P_PC1": pcs[0],
        "T_PC1": pcs[1],
        "F_PC1": pcs[2],
        "recon_err": recon_err,
        "threshold": threshold,
        "decision": decision
    })

# ================= SAVE =================

pd.DataFrame(sensor_log).to_csv("sensor_log.csv", index=False)
pd.DataFrame(system_log).to_csv("system_log.csv", index=False)

print("DONE — pipeline ran successfully.")
