#!/usr/bin/env python3
"""
Streaming Sensor Analytics (PTF + D sensors) with online PCA + peer-deviation + global anomaly checks.

- Reads sensor data from a CSV in data/ matching patterns like 2025...GMT+5x30...csv
- Splits columns into Pressure (P), Temperature (T), Flow (F) sensors using
- MAIN_SENSORS_FILTER. Supports an optional D (diagnostic) group.
- Builds per-group streaming PCA (IncrementalPCA) to extract principal components
- and to enable lightweight reconstruction-based checks.
- Performs peer-to-peer deviation checks within each sensor group (row-wise intra-group checks).
- Uses an online regression model (SGDRegressor) trained from operational sensors to
- derive a dynamic threshold for global deviations (e.g., leaks or operational changes).
- A global reconstruction check (online PCA on all sensors) provides an overall anomaly view.
- Each sensor is assigned a tag: COLD_start, usable, malfunction, etc.
- This is a compact, self-contained script intended to be used as a starting point; adapt
- paths, thresholds and models as needed for your environment.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.linear_model import SGDRegressor
except Exception:
    print("Required scikit-learn not available. Install it to run app_final.py.")
    raise

# Sensor group filters (as provided by the user)
MAIN_SENSORS_FILTER = {
    "P": ["PI", "PIC"],
    "T": ["TI", "TIC"],
    "F": ["FI", "FIC"],
    "D": ["DI"],
}

OPERATIONAL_SENSORS_FILTER = [
    "MOV",
    "PUMP",
    "SCR",
    "XXI",
    "DRA",
    "SBV",
    "TYPE",
    "MP",
]

# Simple thresholds and settings (tune as needed)
PEER_DEVIATION_Z = 2.5
INIT_FRAMES = 40  # number of initial rows to label sensors as COLD_start
MALFUNCTION_CONSEC_FRAMES = 5  # consecutive deviating rows to mark malfunction
GLOBAL_N_COMPONENTS = 4  # components for the global PCA (limited by sensor count)


def compute_peer_dev_chunk(
    chunk: pd.DataFrame, group_cols: List[str], z_thresh: float = PEER_DEVIATION_Z
) -> List[Dict[str, str]]:
    """
    Compute per-row peer deviation within a sensor group.
    Returns a list of dicts mapping column -> status ("usable" or "deviates").
    """
    if len(group_cols) == 0:
        return [{} for _ in range(len(chunk))]
    try:
        X = chunk[group_cols].to_numpy(dtype=float)
    except Exception:
        X = (
            chunk[group_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
    if X.size == 0:
        return [{} for _ in range(len(chunk))]
    means = np.nanmean(X, axis=1)
    stds = X.std(axis=1, ddof=0)
    stds = np.where(stds == 0, np.nan, stds)
    z = (X - means[:, None]) / stds[:, None]
    per_row = []
    for i in range(len(chunk)):
        row_map = {}
        for j, col in enumerate(group_cols):
            val = z[i, j]
            if np.isnan(val):
                row_map[col] = "usable"
            elif abs(float(val)) > z_thresh:
                row_map[col] = "deviates"
            else:
                row_map[col] = "usable"
        per_row.append(row_map)
    return per_row


def locate_csv(data_dir: str = "data") -> str:
    # Try a few patterns that resemble the user's description
    patterns = [
        os.path.join(data_dir, "2025*.csv"),
        os.path.join(data_dir, "*GMT+5*.csv"),
        os.path.join(data_dir, "*GMT+5x*.csv"),
        os.path.join(data_dir, "*.csv"),
    ]
    candidates = []
    for pat in patterns:
        matches = glob.glob(pat)
        candidates.extend(matches)
    if not candidates:
        return ""
    # Return the most recently modified file
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def determine_sensor_groups(columns: List[str]):
    groups: Dict[str, List[str]] = {"P": [], "T": [], "F": [], "D": []}
    op_set = set()
    for col in columns:
        assigned = False
        for g, tags in MAIN_SENSORS_FILTER.items():
            if any(tag and tag in col for tag in tags):
                groups[g].append(col)
                assigned = True
                break
        if not assigned:
            # If not matched by main tags, check for operational tags explicitly
            if any(op in col for op in OPERATIONAL_SENSORS_FILTER):
                op_set.add(col)
    # Deduplicate while preserving order
    for k in list(groups.keys()):
        seen = set()
        unique: List[str] = []
        for c in groups[k]:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        groups[k] = unique
    op_cols = sorted(list(op_set))
    return groups, op_cols


class GroupPCA:
    def __init__(self, cols: List[str], n_components: int = 2):
        self.cols = cols
        self.n_components = max(1, min(n_components, len(cols))) if cols else 0
        self.ipca = (
            IncrementalPCA(n_components=self.n_components)
            if self.n_components > 0
            else None
        )
        self.fitted = False

    def partial_fit(self, X: np.ndarray):
        if self.ipca is None or X.size == 0:
            return
        self.ipca.partial_fit(X)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.ipca is None or not self.fitted:
            return None
        return self.ipca.transform(X)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.ipca is None or not self.fitted:
            return None
        return self.ipca.inverse_transform(Y)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Streaming sensor analytics with PCA and peer-deviation checks"
    )
    parser.add_argument(
        "--data",
        dest="data_path",
        default=None,
        help="Path to the CSV data file. If omitted, the script will try to auto-locate under data/",
    )
    parser.add_argument(
        "--chunksize",
        dest="chunksize",
        type=int,
        default=1000,
        help="CSV chunk size (rows per iteration)",
    )
    parser.add_argument(
        "--log", dest="log_path", default=None, help="Optional log file path"
    )
    parser.add_argument(
        "--initframes",
        dest="init_frames",
        type=int,
        default=INIT_FRAMES,
        help="Frames for COLD_start labeling",
    )
    args = parser.parse_args(argv)

    data_path = args.data_path
    if not data_path:
        data_path = locate_csv()
        if not data_path:
            print(
                "No data CSV found. Provide --data or ensure data/ contains a 2025*.csv file."
            )
            return 1
    if not os.path.isfile(data_path):
        print(f"Data file not found: {data_path}")
        return 1
    print(f"Reading sensor data from: {data_path}")

    # Prepare logging (optional)
    log_file = args.log_path
    if log_file:
        log_f = open(log_file, "a", buffering=1)
    else:
        log_f = None

    # Initialize streaming components after reading the header (columns)
    reader = pd.read_csv(data_path, chunksize=args.chunksize, low_memory=False)
    first_chunk = True
    sensor_groups: Dict[str, List[str]] = {}
    groups: Dict[str, List[str]] = {}
    operational_cols: List[str] = []
    all_sensor_cols: List[str] = []
    pca_by_group: Dict[str, GroupPCA] = {}
    global_pca: IncrementalPCA = None
    global_components = 0
    global_fitted = False

    # Global book-keeping for statuses
    sensor_status: Dict[str, Dict[str, object]] = {}

    # Online regression model to threshold global deviations based on operational sensors
    op_model = SGDRegressor(loss="squared_loss", penalty="l2", max_iter=5)
    op_model_initialized = False
    residual_history: List[float] = []
    residuals_window = 32  # number of residuals to keep for thresholding

    # Helper to log in a consistent way
    def log(msg: str):
        ts = pd.Timestamp.now().isoformat()
        line = f"[{ts}] {msg}"
        print(line)
        if log_f:
            log_f.write(line + "\n")

    groups = {}
    for chunk in reader:
        if first_chunk:
            columns = list(chunk.columns)
            groups, op_cols = determine_sensor_groups(columns)
            operational_cols = op_cols
            # Collect all sensor columns
            all_sensor_cols = []
            for g in ["P", "T", "F", "D"]:
                for c in groups.get(g, []):
                    if c not in all_sensor_cols:
                        all_sensor_cols.append(c)
            # Initialize per-group PCAs
            for g in ["P", "T", "F", "D"]:
                cols = groups.get(g, [])
                if len(cols) == 0:
                    continue
                n_comp = max(1, min(2, len(cols)))  # 1 or 2 components per group
                pca_by_group[g] = GroupPCA(cols, n_components=n_comp)
            # Initialize global PCA across all sensors in this view
            global_components = min(
                len(all_sensor_cols) if all_sensor_cols else 1, GLOBAL_N_COMPONENTS
            )
            if global_components <= 0:
                global_components = 1
            global_pca = IncrementalPCA(n_components=global_components)
            global_fitted = False
            # Initialize sensor_status for all sensors
            for c in all_sensor_cols:
                sensor_status[c] = {
                    "label": "COLD_start",
                    "remaining": args.init_frames,
                    "consecutive": 0,
                }
            first_chunk = False
            log("Initialized sensor groups and PCA models.")

        # Ensure numeric conversion for sensors; convert non-numeric to NaN then to 0
        for c in all_sensor_cols:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            else:
                # If a column disappears in a chunk (unlikely), fill with zeros
                chunk[c] = 0
        # Fill missing columns in chunk with zeros if they exist in sensor list but missing in this chunk
        for c in all_sensor_cols:
            if c not in chunk.columns:
                chunk[c] = 0
        # Reorder columns to maintain consistency
        chunk = chunk[all_sensor_cols]

        # Update per-group PCA with this chunk (fit/partial_fit)
        for g, pca in pca_by_group.items():
            cols = pca.cols
            if len(cols) == 0:
                continue
            X = chunk[cols].to_numpy(dtype=float)
            if X.size == 0:
                continue
            pca.partial_fit(X)
        # After potential initial fit, compute transformed data for all groups
        transformed_by_group = {}
        for g, pca in pca_by_group.items():
            cols = pca.cols
            if len(cols) == 0:
                continue
            X = chunk[cols].to_numpy(dtype=float)
            if not pca.fitted:
                transformed_by_group[g] = None
                continue
            transformed_by_group[g] = pca.transform(X)
        # Reconstruct per-row data for group-level reconstruction error (optional)
        recon_errors = []  # per-row mean squared error across sensors in this chunk
        for g, pca in list(pca_by_group.items()):
            cols = pca.cols
            if len(cols) == 0:
                continue
            X = chunk[cols].to_numpy(dtype=float)
            if X.size == 0:
                continue
            if not pca.fitted:
                transformed_by_group[g] = None
                continue
            # Use the transformed data to reconstruct
            transformed = transformed_by_group.get(g)
            if transformed is None:
                continue
            X_rec = pca.inverse_transform(transformed)
            try:
                err = np.mean((X - X_rec) ** 2, axis=1)
            except Exception:
                err = np.zeros(len(chunk))
            recon_errors.append(err)
        # Combine recon_errors if more than one group contributed
        if recon_errors:
            row_recon_errors = np.mean(np.vstack(recon_errors), axis=0)
        else:
            row_recon_errors = np.zeros(len(chunk))

        # Global PCA across all sensors (for a reconstruction-based global check)
        X_all = chunk[all_sensor_cols].to_numpy(dtype=float)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        if X_all.size > 0:
            global_pca.partial_fit(X_all)
            Z = global_pca.transform(X_all)
            X_all_rec = global_pca.inverse_transform(Z)
            global_row_errors = np.mean((X_all - X_all_rec) ** 2, axis=1)
            # Initialize rolling log for global anomaly threshold if needed
            if len(residual_history) == 0:
                residual_history.extend(global_row_errors.tolist())
            # Append and trim
            residual_history = (residual_history + global_row_errors.tolist())[
                -residuals_window:
            ]
        else:
            global_row_errors = np.zeros(len(chunk))

        # Online operation sensor regression to derive dynamic threshold for global deviations
        if len(operational_cols) > 0:
            X_op = chunk[operational_cols].to_numpy(dtype=float)
            # Target: mean of Pressure sensors as a simple proxy
            if len(all_sensor_cols) > 0 and len(groups.get("P", [])) > 0:
                y_target = chunk[groups.get("P", [])].mean(axis=1).to_numpy(dtype=float)
            else:
                y_target = np.zeros(len(chunk))
            if not op_model_initialized:
                try:
                    op_model.partial_fit(X_op, y_target)
                    op_model_initialized = True
                except Exception:
                    op_model_initialized = True  # avoid retrying forever
            else:
                try:
                    op_model.partial_fit(X_op, y_target)
                except Exception:
                    pass
            try:
                y_pred = op_model.predict(X_op)
                residuals = np.abs(y_target - y_pred)
                for r in residuals:
                    residual_history.append(float(r))
            except Exception:
                residuals = np.zeros(len(chunk))
        else:
            residuals = np.zeros(len(chunk))

        # Sliding threshold for global deviations based on residual history
        if len(residual_history) >= residuals_window:
            recent = np.asarray(residual_history[-residuals_window:])
            mean_res = float(np.mean(recent))
            std_res = (
                float(np.std(recent, ddof=0)) if np.std(recent, ddof=0) > 0 else 0.0
            )
            global_threshold = mean_res + (3.0 * std_res if std_res > 0 else 0.0)
        else:
            global_threshold = float("inf")  # no threshold yet

        # Peer-deviation labeling per row across all sensors
        # Compute per-group deviations and update per-sensor statuses
        if "compute_peer_dev_chunk" in globals():
            per_row_dev_P = compute_peer_dev_chunk(
                chunk, groups.get("P", []), z_thresh=PEER_DEVIATION_Z
            )
            per_row_dev_T = compute_peer_dev_chunk(
                chunk, groups.get("T", []), z_thresh=PEER_DEVIATION_Z
            )
            per_row_dev_F = compute_peer_dev_chunk(
                chunk, groups.get("F", []), z_thresh=PEER_DEVIATION_Z
            )
        else:
            per_row_dev_P = [{} for _ in range(len(chunk))]
            per_row_dev_T = [{} for _ in range(len(chunk))]
            per_row_dev_F = [{} for _ in range(len(chunk))]
        row_count = len(chunk)
        for i in range(row_count):
            # Initialize status for any new sensors already done earlier
            row_status: Dict[str, str] = {}
            for col in all_sensor_cols:
                tag = None
                # Gather tag from any group if present
                if (
                    col in groups.get("P", [])
                    and per_row_dev_P[i].get(col) == "deviates"
                ):
                    tag = "deviates"
                if (
                    col in groups.get("T", [])
                    and per_row_dev_T[i].get(col) == "deviates"
                ):
                    tag = "deviates"
                if (
                    col in groups.get("F", [])
                    and per_row_dev_F[i].get(col) == "deviates"
                ):
                    tag = "deviates"
                if tag:
                    row_status[col] = tag
            # Update statuses based on this row
            for col in all_sensor_cols:
                s = sensor_status.get(
                    col,
                    {
                        "label": "COLD_start",
                        "remaining": args.init_frames,
                        "consecutive": 0,
                    },
                )
                if s["remaining"] > 0:
                    s["remaining"] -= 1
                    sensor_status[col] = s
                    continue
                if col in row_status and row_status[col] == "deviates":
                    s["consecutive"] += 1
                    if s["consecutive"] >= MALFUNCTION_CONSEC_FRAMES:
                        s["label"] = "malfunction"
                else:
                    s["consecutive"] = 0
                    if s["label"] != "malfunction":
                        s["label"] = "usable"
                sensor_status[col] = s

        # Log a concise summary for this chunk
        counts = {"COLD_start": 0, "malfunction": 0, "usable": 0, "deviates": 0}
        for c in all_sensor_cols:
            tag = sensor_status[c]["label"]
            counts[tag] = counts.get(tag, 0) + 1
        log(
            f"Chunk processed: rows={len(chunk)} | statuses: COLD_start={counts.get('COLD_start', 0)} "
            f"malfunction={counts.get('malfunction', 0)} usable={counts.get('usable', 0)}"
        )

        # Optional: log global anomaly flags per row
        global_anomalies = global_row_errors > (
            global_threshold if global_threshold != float("inf") else float("inf")
        )
        if isinstance(global_anomalies, np.ndarray) and global_anomalies.any():
            n = int(np.sum(global_anomalies))
            log(
                f"Global anomaly detected in {n} of {len(chunk)} rows (threshold={global_threshold:.4f})."
            )

        # Flush log to disk if configured
        if log_f and not log_f.closed:
            log_f.flush()

    if log_f:
        log_f.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
