# -*- coding: utf-8 -*-
"""
TE-Data-Process: Tennessee Eastman (TE) 工艺异常检测基线脚本

功能：
- 加载 ./data/ 目录下的 d00_te.dat (正常) 及 d01_te.dat..d21_te.dat (故障) 数据
- 探索数据（shape、缺失值）
- 提供两种检测器：PCA 重建误差 或 Autoencoder 重建误差（可选 IsolationForest）
- 在正常数据上确定阈值（百分位法或均值+k*std）
- 对每个故障计算样本级检测率（关注样本 index 161-960）
- 输出检测率表、若干时间线示例图、总体度量（平均检测率等）
- 保存图像到 output/ 目录

使用：
python te_detection.py --method pca
python te_detection.py --method autoencoder --threshold_percentile 99

依赖：numpy, pandas, scikit-learn, matplotlib, tensorflow (若使用 autoencoder)
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# optionally import keras if autoencoder chosen
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

DATA_DIR = "./data"
OUTPUT_DIR = "./output"
FAULT_START_IDX = 161  # 包括该索引：评估使用 samples 161..960 (0-based)
TRAIN_FILENAME = "d00_te.dat"
FAULT_PATTERN = "d??_te.dat"  # includes d00..d21

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_files(data_dir=DATA_DIR):
    """加载 data_dir 中以 _te.dat 结尾的文件，返回 dict filename->np.array"""
    files = sorted(glob.glob(os.path.join(data_dir, "*_te.dat")))
    data = {}
    for f in files:
        name = os.path.basename(f)
        try:
            df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
            arr = df.values
            data[name] = arr
        except Exception as e:
            print(f"无法读取 {f}: {e}")
    return data

def explore_data(data_dict):
    """打印并返回一些探索性信息"""
    print("已加载文件：")
    for k,v in data_dict.items():
        print(f" - {k}: shape={v.shape}, NaN={np.isnan(v).sum()}")
    # 检查 d00 存在
    if TRAIN_FILENAME not in data_dict:
        raise FileNotFoundError(f"{TRAIN_FILENAME} 未找到，请确认数据文件存在于 {DATA_DIR}")
    # 简单统计
    train = data_dict[TRAIN_FILENAME]
    print(f"\n正常数据（{TRAIN_FILENAME}）样本数: {train.shape[0]}, 特征数: {train.shape[1]}")
    print("前5行示例：")
    print(pd.DataFrame(train).head())
    return

def fit_pca_detector(X_train, n_components=10):
    """使用 PCA 重建误差作为异常分数。返回 scaler, pca 模型"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=0)
    Xproj = pca.fit_transform(Xs)
    Xrec = pca.inverse_transform(Xproj)
    rec_err = np.mean((Xs - Xrec)**2, axis=1)
    return {"scaler": scaler, "pca": pca, "train_rec_err": rec_err}

def build_autoencoder(input_dim, encoding_dim=10):
    """构建一个简单的全连接 Autoencoder"""
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_autoencoder_detector(X_train, encoding_dim=10, epochs=50, batch_size=64, verbose=0):
    """训练 AE，返回 scaler, model, 训练重构误差"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = build_autoencoder(input_dim=Xs.shape[1], encoding_dim=encoding_dim)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(Xs, Xs, validation_split=0.1, epochs=epochs, batch_size=batch_size,
                        callbacks=[es], verbose=verbose)
    Xrec = model.predict(Xs)
    rec_err = np.mean((Xs - Xrec)**2, axis=1)
    return {"scaler": scaler, "ae": model, "train_rec_err": rec_err, "history": history}

def determine_threshold(rec_err, method="percentile", percentile=99, k=3):
    """根据训练重构误差确定阈值"""
    if method == "percentile":
        thr = np.percentile(rec_err, percentile)
    else:
        thr = rec_err.mean() + k * rec_err.std()
    return thr

def eval_on_file(arr, detector, method="pca", thr=None):
    """对单个数据文件计算每个样本的重构误差及是否为异常（bool数组）"""
    X = arr.copy()
    if method == "pca":
        scaler = detector["scaler"]
        pca = detector["pca"]
        Xs = scaler.transform(X)
        Xproj = pca.transform(Xs)
        Xrec = pca.inverse_transform(Xproj)
        rec_err = np.mean((Xs - Xrec)**2, axis=1)
    elif method == "autoencoder":
        scaler = detector["scaler"]
        ae = detector["ae"]
        Xs = scaler.transform(X)
        Xrec = ae.predict(Xs)
        rec_err = np.mean((Xs - Xrec)**2, axis=1)
    else:
        raise ValueError("未知 method")
    alarms = rec_err > thr
    return rec_err, alarms

def compute_detection_rates(data_dict, detector, method, thr):
    """对 d01-d21 分别计算检测率、首次告警位置、检测延迟等"""
    results = []
    for fname, arr in sorted(data_dict.items()):
        if not fname.startswith("d") or fname == TRAIN_FILENAME:
            continue
        # 只关注评估区间 161..960
        rec_err, alarms = eval_on_file(arr, detector, method=method, thr=thr)
        eval_idx = np.arange(161, arr.shape[0])  # inclusive 161..960
        eval_alarms = alarms[eval_idx]
        num_fault_samples = len(eval_idx)
        detected_count = int(np.sum(eval_alarms))
        detection_rate = detected_count / num_fault_samples
        # 首次检测位置（相对于文件内索引），如果未检测到则设为 None
        detected_positions = eval_idx[eval_alarms]
        first_detect = int(detected_positions[0]) if len(detected_positions) > 0 else None
        delay = first_detect - 161 + 1 if first_detect is not None else None  # 以样本数表示的延迟
        results.append({
            "fault_file": fname,
            "fault_id": fname.split("_")[0],
            "detection_rate": detection_rate,
            "detected_count": detected_count,
            "num_fault_samples": num_fault_samples,
            "first_detect_idx": first_detect,
            "delay_samples": delay
        })
    df = pd.DataFrame(results).sort_values("fault_file").reset_index(drop=True)
    return df

def plot_detection_timeline(arr, rec_err, alarms, fname, fault_start=161, savepath=None):
    """为单个文件画时间线：重构误差与告警点"""
    plt.figure(figsize=(12,4))
    x = np.arange(arr.shape[0])
    plt.plot(x, rec_err, label='reconstruction error')
    plt.plot(x, alarms.astype(int)*rec_err.max()*1.05, 'r|', label='alarm markers')
    plt.axvline(fault_start, color='k', linestyle='--', label='fault start')
    plt.title(f"Detection timeline: {fname}")
    plt.xlabel("sample index")
    plt.ylabel("reconstruction error")
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    plt.close()

def main(args):
    data = load_all_files(DATA_DIR)
    explore_data(data)
    # 准备训练数据（d00）
    train = data[TRAIN_FILENAME]
    X_train = train  # use full training samples (0..960)
    print("\n训练样本数量:", X_train.shape)

    if args.method == "pca":
        det = fit_pca_detector(X_train, n_components=args.n_components)
        rec_err_train = det["train_rec_err"]
    elif args.method == "autoencoder":
        if not KERAS_AVAILABLE:
            raise RuntimeError("Autoencoder 需要安装 tensorflow/keras")
        det = fit_autoencoder_detector(X_train, encoding_dim=args.encoding_dim,
                                       epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)
        rec_err_train = det["train_rec_err"]
    else:
        raise ValueError("未知 method")

    thr = determine_threshold(rec_err_train, method=args.threshold_method,
                              percentile=args.threshold_percentile, k=args.threshold_k)
    print(f"\n使用阈值: {thr:.6g} (方法={args.threshold_method})")

    # 评估
    df_res = compute_detection_rates(data, det, method=args.method, thr=thr)
    print("\n检测结果（每个故障）:")
    print(df_res[["fault_file","detection_rate","first_detect_idx","delay_samples"]])

    # 总体指标
    mean_detection_rate = df_res["detection_rate"].mean()
    print(f"\n平均检测率 (21 faults): {mean_detection_rate:.4f}")

    # 保存表
    out_table = os.path.join(OUTPUT_DIR, f"detection_results_{args.method}.csv")
    df_res.to_csv(out_table, index=False)
    print(f"检测结果表已保存: {out_table}")

    # 可视化整体条形图
    plt.figure(figsize=(10,5))
    plt.bar(df_res["fault_id"], df_res["detection_rate"])
    plt.xlabel("fault id")
    plt.ylabel("detection rate")
    plt.title(f"Detection rate per fault ({args.method})")
    plt.ylim(0,1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(OUTPUT_DIR, f"detection_rates_{args.method}.png"), bbox_inches='tight')
    plt.show()
    plt.close()

    # 画几个示例时间线（选择检测率最高/最低/中等的故障）
    df_sorted = df_res.sort_values("detection_rate")
    sample_faults = []
    if len(df_sorted) >= 3:
        sample_faults = [
            df_sorted.iloc[0]["fault_file"],  # 最差
            df_sorted.iloc[len(df_sorted)//2]["fault_file"],  # 中位
            df_sorted.iloc[-1]["fault_file"]  # 最好
        ]
    else:
        sample_faults = df_sorted["fault_file"].tolist()

    for fname in sample_faults:
        arr = data[fname]
        rec_err, alarms = eval_on_file(arr, det, method=args.method, thr=thr)
        plot_detection_timeline(arr, rec_err, alarms, fname,
                                fault_start=FAULT_START_IDX,
                                savepath=os.path.join(OUTPUT_DIR, f"timeline_{fname}_{args.method}.png"))

    print("\n输出文件夹内容：", os.listdir(OUTPUT_DIR))
    print("完成。请检查 output/ 目录中的 csv 与 png 文件用于后续分析。")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["pca","autoencoder"], default="pca", help="检测方法，pca 或 autoencoder")
    p.add_argument("--n_components", type=int, default=10, help="PCA 主成分数")
    p.add_argument("--encoding_dim", type=int, default=10, help="Autoencoder 压缩维度")
    p.add_argument("--epochs", type=int, default=100, help="AE 训练轮数")
    p.add_argument("--batch_size", type=int, default=64, help="AE 批大小")
    p.add_argument("--threshold_method", choices=["percentile","mean_std"], default="percentile")
    p.add_argument("--threshold_percentile", type=float, default=99.0)
    p.add_argument("--threshold_k", type=float, default=3.0)
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()
    main(args)