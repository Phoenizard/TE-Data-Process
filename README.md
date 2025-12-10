# TE-Data-Process — Tennessee Eastman 异常检测（PCA baseline）

项目简介
--------
本仓库实现了对 Tennessee Eastman (TE) 工艺数据的故障检测基线流程（基于 PCA 重构误差）。数据组织遵循常见 TEbench 格式：`./data/` 目录下包含 `d00_te.dat`（正常）与 `d01_te.dat` 至 `d21_te.dat`（21 种故障），每个文件为 961×52 的空格分隔文本（0..960 行，样本 161 开始注入故障）。采样间隔 3 分钟。

已生成结果（在 main 分支 output/）
- output/detection_results_pca.csv — 每个故障的检测率、首次告警位置、延迟等表格（CSV）
- output/detection_rates_pca.png — 各故障检测率条形图
- output/timeline_d06_te.dat_pca.png — 示例故障时间线（重构误差 + 告警）
- output/timeline_d09_te.dat_pca.png
- output/timeline_d18_te.dat_pca.png

运行说明
--------
1. 环境（推荐使用虚拟环境）
   - Python 3.8+
   - 依赖请见 `requirements.txt`（numpy, pandas, scikit-learn, matplotlib，若使用 Autoencoder 则需 tensorflow）

2. 使用方法（PCA baseline）
   - 将 `./data/` 放置 TE 数据文件（d00_te.dat, d01_te.dat ... d21_te.dat）
   - 运行脚本（仓库根）：
     python te_detection.py --method pca --n_components 10 --threshold_percentile 99
   - 输出保存在 `./output/` 下

3. 输出说明
   - detection_results_pca.csv：包含每个故障的 detection_rate（针对样本 161..960）、首次检测索引、延迟（样本数）
   - detection_rates_pca.png：整体检测率可视化
   - timeline_*.png：若干故障的重构误差时间线和告警标记

简要结论（基线）
----------------
- 本次 PCA 基线在 21 个故障上的平均检测率约为 63.84%（见实验报告中的完整表格）。
- 部分故障检测效果很好（例如 d01, d02, d04, d06, d07, d14 等接近或等于 1.0），但也存在若干检测率极低或延迟较大的故障（例如 d03、d09、d15 检测率均非常低）。
- 详细实验与分析参阅仓库中的实验报告（EXPERIMENT_REPORT.md）。

许可与作者
-----------
- 作者：项目贡献者
- 本仓库用于研究与实验交流，代码与报告仅供参考。