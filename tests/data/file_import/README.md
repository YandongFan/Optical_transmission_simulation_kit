# 文件导入回归样本

本目录用于“相位分布/透射率分布”文件导入回归验证。

## 已内置样本（可直接回归）

已提供 3 组不同维度 CSV 样本（每组 `phase + trans`）：

- `case_a_4x4_phase.csv` / `case_a_4x4_trans.csv`
- `case_b_3x5_phase.csv` / `case_b_3x5_trans.csv`
- `case_c_2x6_phase.csv` / `case_c_2x6_trans.csv`

## 可选生成脚本（MAT v7.3）

运行（需要本地 Python 环境具备 `h5py`）：

```bash
py scripts/generate_file_import_test_data.py
```

将额外生成 3 组不同维度 `.mat` 样本（每组包含 `phase.mat` + `trans.mat`）：

- `case_a_64x64_*`
- `case_b_128x32_*`
- `case_c_45x91_*`

## 预期预览截图

请在本地 GUI 手动导入后保存截图到 `tests/data/file_import/screenshots/`：

- `case_a_phase.png`
- `case_a_trans.png`
- `case_b_phase.png`
- `case_b_trans.png`
- `case_c_phase.png`
- `case_c_trans.png`

说明：自动化环境无法直接启动完整桌面 GUI 截图流程，因此截图由本地回归流程产出。
