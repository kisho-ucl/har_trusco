import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timezone, timedelta
from scipy import stats

# ------------------------------------
# IDマッピング
# ------------------------------------
id_dict = {}
path_dict = '/home/kisho_ucl/kisho_ws/warehouse_task_recognition/data/id_dict.txt'
with open(path_dict, "r") as file:
    for line in file:
        key, value = line.strip().split(": ")
        id_dict[int(key)] = int(value)

subid_dict = {value: key for key, value in id_dict.items()}


# ------------------------------------
# 最大サイズの pickle 読み込み
# ------------------------------------
def load_largest_data(file_paths):
    largest_val = None
    largest_ts = None
    max_size = 0

    for file_path in file_paths:
        with open(file_path, "rb") as f:
            ts, val = pickle.load(f)
            if len(val) > max_size:
                max_size = len(val)
                largest_val = val
                largest_ts = ts

    return largest_ts, largest_val


# ------------------------------------
# 時刻 → UNIX (JST)
# ------------------------------------
def time_to_unixtime_jst(time_str, Year, Date):
    jst = timezone(timedelta(hours=9))
    month = int(str(Date)[:2])
    day   = int(str(Date)[2:])
    hour, minute, second = map(int, time_str.split(':'))
    dt = datetime(Year, month, day, hour, minute, second, tzinfo=jst)
    return int(dt.timestamp())


# ------------------------------------
# ファイル読み込み（オリジナルのみ使用）
# ------------------------------------
def read_files(ID, Year, Date, hour, unlabeled=False):
    sub_ID = subid_dict.get(ID)

    search_pattern = f"*{Year}*{Date}*_{hour}_inertial.pkl"
    base_dir = f"/mnt/bigdata/01_projects/2024_trusco/expt_data/{Year}{Date}/inertial/{sub_ID}/"

    file_paths = glob(os.path.join(base_dir, search_pattern))
    if not file_paths:
        print("No files found.")
        return None, -1

    ts, val = load_largest_data(file_paths)

    if unlabeled:
        return ts, val, pd.DataFrame()

    df_label = pd.read_csv(
        f"/home/kisho_ucl/kisho_ws/deep_HAR/data/label/{Year}{Date}_old/subject_{ID}_{hour}.csv",
        index_col=0
    )
    df_label["unixtime"] = df_label["time"].apply(time_to_unixtime_jst, args=(Year, Date))

    return ts, val, df_label


# ------------------------------------
# IMU 抽出（linacc + gyro）
# ------------------------------------
def extract_imu_data(val):
    linacc = val[:, 0:3] - val[:, 3:6]
    gyro   = val[:, 6:9]
    imu_data = np.hstack([linacc, gyro])   # shape = (N, 6)
    return imu_data


# ------------------------------------
# セッション読み込み → IMU & ラベル配列を返す
# ------------------------------------
def load_session_trusco(ID, Year, Date, Hour, op_filter=True):
    ts, val, df_label = read_files(ID, Year, Date, Hour, unlabeled=False)
    if ts is None:
        return None, None

    imu = extract_imu_data(val)       # (N, 6)
    ts_sec = ts.astype(int)

    # unixtime → label
    label_map = dict(zip(df_label["unixtime"].astype(int), df_label["label"]))
    labels = np.array([label_map.get(t, -1) for t in ts_sec], dtype=int)

    if op_filter:
        valid = np.isin(labels, [0, 1, 2])
        imu = imu[valid]
        labels = labels[valid]

    return imu, labels


# ------------------------------------
# ウィンドウ分割（center or majority）
# ------------------------------------
def window_split_trusco(imu, labels, window_size=256, stride=128, mode="center"):
    windows = []
    win_labels = []

    N = len(labels)

    for start in range(0, N - window_size, stride):
        end = start + window_size
        mid = start + window_size // 2

        win = imu[start:end].T   # (6, window)

        if mode == "center":
            label = labels[mid]
        else:
            label = stats.mode(labels[start:end], keepdims=True).mode[0]

        windows.append(win)
        win_labels.append(label)

    return np.array(windows), np.array(win_labels)
