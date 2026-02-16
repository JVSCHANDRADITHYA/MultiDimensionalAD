import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.animation import Animation
import time
from collections import defaultdict, deque
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

# === MY CONFIG DATA ===
WINDOW = 50 # WINDOW USED FOR PCA NO WINDOWED SCALING DUE TO OUTLIER POSSIBILITIES
PCA_COMPONENTS = 3
COLD_START_CRITERIA = 50 # first 50, are 0 then it's gon

DATA_PATH = r'F:\MultiDimensionalAD\data\202505161524GMT+5x30.h24.csv'

RAW_LOG_FILE = 'log.csv'
SENSOR_STATE_LOG = 'sensor_log.csv'

MAIN_SENSORS_FILTER = {
    "P" : ["PI", "PIC"],
    "T" : ["TI", "TIC"],
    "F" : ["FI", "FIC"],
    "D" : ["DI"]
}

OPERATIONAL_SENSORS_FILTER = [
    "MOV", "PUMP", "SCR", "XXI", "DRA", "SBV", "TYPE", "MP"
]

SENSOR_STATES = [
    "HEALTHY", # GOOD SENSORS [CONStailIDERED FOR NEXT TRAINIGN STEPS]
    "NON-OPERATIONAL", # COLD START SENSORS 
    "DEVIATING", # MIGHT HAVE PROBLEMS, FOUND OUT USING P2P DEVIATION TECH
    "OPERATIONAL_DRIVEN", # MIGHT HAVE NO PROBLEMS, CHANGE INDUCED DUE TO CHANGE IN OPERATIONAL STATES
    "EXCLUDED" # ANOMALOUS SENSORS, MIGHT BE COLD_START OR NON-OPERATIONAL WE DO NTO CONSIDER THIS FOR TRAINING PURPOSES
]
# === CONFIG END ===

main_sensor_list = defaultdict(list)
operational_sensor_list = []

def get_sensor_type(col):
    for type in MAIN_SENSORS_FILTER:
        if any(k in col for k in MAIN_SENSORS_FILTER[type]):
            main_sensor_list[type].append(col)
            
    if any(k in col for k in OPERATIONAL_SENSORS_FILTER):
        operational_sensor_list.append(col)

                
# === Scalers and PCA logics
scaler = StandardScaler()
pca = IncrementalPCA(n_components=PCA_COMPONENTS)


# === PRINT SENSORS LIST ===

df = pd.read_csv(DATA_PATH)
df_sensors = df.drop(columns=['Seconds', 'Timestamp_IST', 'WA-PIC1208PV-S'], axis=1)

for col in df_sensors.columns:
    get_sensor_type(col)

from pprint import pprint

pprint(operational_sensor_list)
print()
pprint(main_sensor_list)

total_sensor_count = len(df_sensors.columns)
print(f"\nTotal Sensors: {total_sensor_count}")
total_sensors_found = sum([len(main_sensor_list[k]) for k in main_sensor_list]) + len(operational_sensor_list)
print(f"Sensors Found in Filters: {total_sensors_found}")
missing_sensors = set(df_sensors.columns) - set([sensor for sensors in main_sensor_list.values() for sensor in sensors]) - set(operational_sensor_list)
print(f"Sensors NOT CONSIDERED: {missing_sensors}")

# === END SENSOR PRINT ===

# === START PLOT ===
fig, ax = plt.subplot_mosaic(
    mosaic=[
        ["A", "B", "C"], # PRESSURE > TEMP > FLOW
        ["F", "F", "D"], # F - PCA reconstrcution , D - recomnstrcution loss
        ["F", "F", "E"]  #  F - PCA reconsturcitons, same as beofer, E - average P, Temp and FLow across the valid sensors
    ], 
    constrained_layout = True
)

ax_pressure = ax["A"]
ax_temp     = ax["B"]
ax_flow     = ax["C"]

ax_pca_rec  = ax["F"]   # PCA components (PC1, PC2, PC3)
ax_pca_loss = ax["D"]   # reconstruction error
ax_avg      = ax["E"]   # avg P / T / F

ax_pressure.set_title("Pressure Sensors (PI / PIC)")
ax_temp.set_title("Temperature Sensors (TI)")
ax_flow.set_title("Flow Sensors (FI / FIC)")

ax_pca_rec.set_title("PCA Components (PC1, PC2, PC3)")
ax_pca_loss.set_title("PCA Reconstruction Loss")
ax_avg.set_title("Average P / T / F (Healthy Sensors)")


for a in [ax_pressure, ax_temp, ax_flow, ax_pca_rec, ax_pca_loss, ax_avg]:
    a.set_xlabel("Time")
    a.set_ylabel("Value")

pressure_lines = {}
temp_lines = {}
flow_lines = {}

pc_lines = {
    "PC1": ax_pca_rec.plot([], [], label="PC1")[0],
    "PC2": ax_pca_rec.plot([], [], label="PC2")[0],
    "PC3": ax_pca_rec.plot([], [], label="PC3")[0],
}
pca_threshold_line = ax_pca_rec.axhline(
    0, linestyle="--", color="red", label="Dynamic Threshold"
)

loss_line = ax_pca_loss.plot([], [], color="black", label="Recon Loss")[0]


avg_p_line = ax_avg.plot([], [], label="Avg Pressure")[0]
avg_t_line = ax_avg.plot([], [], label="Avg Temperature")[0]
avg_f_line = ax_avg.plot([], [], label="Avg Flow")[0]


ax_pca_rec.legend()
ax_pca_loss.legend()
ax_avg.legend()


# === PLOT FINISH ===

# === STREAMING BUFFERS ===

sensor_buffers = defaultdict(lambda: deque(maxlen=WINDOW))
sensor_states = {}

# PCA buffers
pca_input_buffer = deque(maxlen=WINDOW)
pca_scores_buffer = deque(maxlen=WINDOW)
pca_recon_err_buffer = deque(maxlen=WINDOW)

# Averages
avg_p_buf = deque(maxlen=WINDOW)
avg_t_buf = deque(maxlen=WINDOW)
avg_f_buf = deque(maxlen=WINDOW)

# Operational signal (scalar energy proxy)
operational_energy_buf = deque(maxlen=WINDOW)

# Init states
for col in df_sensors.columns:
    sensor_states[col] = "HEALTHY"

def window_x(n, t):
    return np.arange(t - n + 1, t + 1)

def compute_operational_energy(row):
    vals = []
    for s in operational_sensor_list:
        v = row.get(s, np.nan)
        if np.isfinite(v):
            vals.append(abs(v))
    return np.mean(vals) if vals else 0.0

alpha = 0.0
beta = 0.0
REG_MIN_SAMPLES = 30

def update_threshold_regression(op_buf, err_buf):
    global alpha, beta

    n = min(len(op_buf), len(err_buf))
    if n < REG_MIN_SAMPLES:
        return

    X = np.array(op_buf)[-n:]
    y = np.array(err_buf)[-n:]

    if np.std(X) < 1e-6:
        return

    Xc = X - X.mean()
    yc = y - y.mean()

    denom = np.dot(Xc, Xc)
    if denom < 1e-6:
        return

    alpha = np.dot(Xc, yc) / denom
    beta = y.mean() - alpha * X.mean()

for col in main_sensor_list["P"]:
    pressure_lines[col] = ax_pressure.plot([], [], linewidth=0.8)[0]

for col in main_sensor_list["T"]:
    temp_lines[col] = ax_temp.plot([], [], linewidth=0.8)[0]

for col in main_sensor_list["F"]:
    flow_lines[col] = ax_flow.plot([], [], linewidth=0.8)[0]



mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.ion()
plt.show()
