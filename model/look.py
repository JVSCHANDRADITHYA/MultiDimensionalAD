import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
WINDOW = 50
DATA_PATH =r"F:\MultiDimensionalAD\data\long_run_varying_flows.csv" #r'F:\MultiDimensionalAD\data\202505161524GMT+5x30.h24.csv'

MAIN_SENSORS_FILTER = {
    "P": ["PI", "PIC", "PT"],
    "T": ["TI", "TIC", "TT"],
    "F": ["FI", "FIC"],
}

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
df_sensors = df.drop(
    columns=['Seconds', 'Timestamp_IST', 'WA-PIC1208PV-S'],
    errors="ignore"
)

# ================= SENSOR GROUPING =================
main_sensor_list = defaultdict(list)

for col in df_sensors.columns:
    for k, tags in MAIN_SENSORS_FILTER.items():
        if any(t in col for t in tags):
            main_sensor_list[k].append(col)

print("Pressure:", main_sensor_list["P"])
print("Temperature:", main_sensor_list["T"])
print("Flow:", main_sensor_list["F"])

# ================= PLOT SETUP =================
plt.ion()

fig, ax = plt.subplot_mosaic(
    [
        ["A", "B", "C"],
        ["D", "E", "E"],
    ],
    constrained_layout=True,
    figsize=(15, 9)
)

ax_pressure = ax["A"]
ax_temp = ax["B"]
ax_flow = ax["C"]

ax_avg = ax["D"]        # averages
ax_pca = ax["E"]        # PCA per group

ax_pressure.set_title("Pressure Sensors")
ax_temp.set_title("Temperature Sensors")
ax_flow.set_title("Flow Sensors")

ax_avg.set_title("Average P / T / F")
ax_pca.set_title("PCA (PC1) per Group")

for a in [ax_pressure, ax_temp, ax_flow, ax_avg, ax_pca]:
    a.set_xlabel("Time")
    a.set_ylabel("Value")
    a.grid(True)

# ================= LINES =================
pressure_lines = {}
temp_lines = {}
flow_lines = {}

for col in main_sensor_list["P"]:
    pressure_lines[col] = ax_pressure.plot([], [], linewidth=0.7)[0]

for col in main_sensor_list["T"]:
    temp_lines[col] = ax_temp.plot([], [], linewidth=0.7)[0]

for col in main_sensor_list["F"]:
    flow_lines[col] = ax_flow.plot([], [], linewidth=0.7)[0]

# ---- average lines ----
avg_p_line = ax_avg.plot([], [], label="Avg Pressure", linewidth=2)[0]
avg_t_line = ax_avg.plot([], [], label="Avg Temperature", linewidth=2)[0]
avg_f_line = ax_avg.plot([], [], label="Avg Flow", linewidth=2)[0]
ax_avg.legend()

# ---- PCA lines ----
pca_p_line = ax_pca.plot([], [], label="Pressure PC1", linewidth=2)[0]
pca_t_line = ax_pca.plot([], [], label="Temperature PC1", linewidth=2)[0]
pca_f_line = ax_pca.plot([], [], label="Flow PC1", linewidth=2)[0]
ax_pca.legend()

# ================= BUFFERS =================
sensor_buffers = defaultdict(lambda: deque(maxlen=WINDOW))

avg_p_buf = deque(maxlen=WINDOW)
avg_t_buf = deque(maxlen=WINDOW)
avg_f_buf = deque(maxlen=WINDOW)

pca_p_buf = deque(maxlen=WINDOW)
pca_t_buf = deque(maxlen=WINDOW)
pca_f_buf = deque(maxlen=WINDOW)

scaler = StandardScaler()

t_global = 0

def window_x(n, t):
    return np.arange(t - n + 1, t + 1)

# ================= STREAM LOOP =================
plt.show()

for _, row in df_sensors.iterrows():

    # ---- append sensor values ----
    for col in main_sensor_list["P"] + main_sensor_list["T"] + main_sensor_list["F"]:
        v = row[col]
        if np.isfinite(v):
            sensor_buffers[col].append(v)

    # ---- averages ----
    if main_sensor_list["P"]:
        avg_p_buf.append(np.mean([sensor_buffers[c][-1] for c in main_sensor_list["P"] if sensor_buffers[c]]))
    if main_sensor_list["T"]:
        avg_t_buf.append(np.mean([sensor_buffers[c][-1] for c in main_sensor_list["T"] if sensor_buffers[c]]))
    if main_sensor_list["F"]:
        avg_f_buf.append(np.mean([sensor_buffers[c][-1] for c in main_sensor_list["F"] if sensor_buffers[c]]))

    # ---- PCA per group (PC1) ----
    def compute_pc1(cols, buf):
        if len(cols) < 2:
            return
        mat = np.array([sensor_buffers[c] for c in cols if len(sensor_buffers[c]) == WINDOW])
        if mat.shape[0] < 2:
            return
        X = mat.T
        Xs = scaler.fit_transform(X)
        pc1 = PCA(n_components=1).fit_transform(Xs)[-1, 0]
        buf.append(pc1)

    compute_pc1(main_sensor_list["P"], pca_p_buf)
    compute_pc1(main_sensor_list["T"], pca_t_buf)
    compute_pc1(main_sensor_list["F"], pca_f_buf)

    # ---- update sensor plots ----
    for col, line in pressure_lines.items():
        buf = sensor_buffers[col]
        if len(buf) > 1:
            line.set_data(window_x(len(buf), t_global), buf)

    for col, line in temp_lines.items():
        buf = sensor_buffers[col]
        if len(buf) > 1:
            line.set_data(window_x(len(buf), t_global), buf)

    for col, line in flow_lines.items():
        buf = sensor_buffers[col]
        if len(buf) > 1:
            line.set_data(window_x(len(buf), t_global), buf)

    # ---- update averages ----
    if avg_p_buf:
        avg_p_line.set_data(window_x(len(avg_p_buf), t_global), avg_p_buf)
    if avg_t_buf:
        avg_t_line.set_data(window_x(len(avg_t_buf), t_global), avg_t_buf)
    if avg_f_buf:
        avg_f_line.set_data(window_x(len(avg_f_buf), t_global), avg_f_buf)

    # ---- update PCA plot ----
    if pca_p_buf:
        pca_p_line.set_data(window_x(len(pca_p_buf), t_global), pca_p_buf)
    if pca_t_buf:
        pca_t_line.set_data(window_x(len(pca_t_buf), t_global), pca_t_buf)
    if pca_f_buf:
        pca_f_line.set_data(window_x(len(pca_f_buf), t_global), pca_f_buf)

    # ---- lock x-axis ----
    for a in [ax_pressure, ax_temp, ax_flow, ax_avg, ax_pca]:
        a.set_xlim(t_global - WINDOW + 1, t_global)
        a.relim()
        a.autoscale_view(scalex=False, scaley=True)

    t_global += 1
    plt.pause(0.01)

plt.ioff()
plt.show()
