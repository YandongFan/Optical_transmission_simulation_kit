# 光学仿真套件 (Optical Transmission Simulation Kit)

这是一个用于光学传输仿真的完整套件，支持图形化界面、GPU 加速和数据可视化。

## 功能特性 (Features)
- **参数配置**: 支持光路方向、三维网格、波长等参数设置。
- **光源模拟**: 内置平面波、高斯光束、拉盖尔-高斯光束、贝塞尔光束。
- **调制平面**: 支持自定义相位和透射率调制，支持导入 CSV/MAT 文件，支持多边形几何掩膜定义。
- **传播算法**: 采用 GPU 加速的角谱法 (Angular Spectrum Method)。
- **监视器系统**: 实时显示光场强度和相位分布，支持数据导出 (HDF5, MAT, CSV)。
- **图形界面**: 基于 PyQt6 的现代化 GUI，集成 Matplotlib 可视化。
  - **动态布局**: 支持监视器列表与可视化区域的拖拽分割与状态持久化。
  - **结果对比**: 支持监视器结果的并排与差值对比。

## 更新日志 (Changelog)

### Fixes & Improvements (2025-03-15)
- **已修复 Z 维度范围截断问题**: 修正了仿真引擎在 YZ/XZ 平面监视器存在时未正确覆盖全 Z 轴范围（如 0~200um）的缺陷。
- **GUI 分割条持久化**: 
  - 实现了“监视器列表”与“可视化结果”区域的垂直拖拽分割。
  - 自动记录分割位置到 `<user_home>/.optical_simulation_kit/gui_layout.json`，下次启动自动恢复。
  - 优化了可视化区域的滚动行为，防止小屏幕下内容溢出。

## 安装指南 (Installation)

### 前置要求 (Prerequisites)
- Python 3.8+
- CUDA Toolkit (如果需要 GPU 加速)

### 安装步骤 (Steps)
1. 克隆代码库:
   ```bash
   git clone https://github.com/your-repo/optical-simulation-kit.git
   cd optical-simulation-kit
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   或者直接安装:
   ```bash
   pip install PyQt6 numpy torch scipy matplotlib h5py pandas pytest pyinstaller
   ```

## 使用说明 (Usage)

### 启动程序 (Run)
运行以下命令启动图形界面:
```bash
python run.py
```

### 操作流程 (Workflow)
1. **网格设置**: 在 "Grid & Direction" 标签页设置网格大小 (Nx, Ny) 和间距 (dx, dy)。
2. **光源配置**: 在 "Source" 标签页选择光源类型并设置参数 (如束腰半径 w0)。
3. **调制器配置**: 在 "Mod 1" 和 "Mod 2" 标签页加载相位或透射率文件 (CSV/MAT)。
4. **预览**: 点击 "Preview" 按钮查看初始光场分布。
5. **运行仿真**: 点击 "Run Simulation" 按钮开始传播计算。
6. **结果查看**: 在右侧可视化面板查看光强和相位分布。

## 文件格式说明 (File Formats)
- **CSV**: 纯文本格式，支持二维数组加载。
- **MAT**: MATLAB 数据格式，默认读取第一个非系统变量。
- **HDF5**: 标准分层数据格式，用于导出完整光场数据。

## 开发与测试 (Development)
运行单元测试:
```bash
python -m pytest tests/test_core.py
```

## 打包 (Packaging)
使用 PyInstaller 生成可执行文件:
```bash
pyinstaller --name "OpticalSim" --windowed --add-data "src;src" run.py
```
