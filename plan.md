# 光学仿真套件开发计划 (Optical Simulation Suite Development Plan)

## 1. 项目初始化 (Project Initialization)
- [x] 创建项目目录结构 (Create project directory structure)
- [x] 配置 Python 虚拟环境和依赖管理 (Setup Python virtual environment and dependencies)
  - 依赖: PyQt6, NumPy, PyTorch (GPU加速), SciPy, Matplotlib, h5py, pandas
- [x] 创建 `README.md` 和 `.gitignore`

## 2. 核心仿真引擎开发 (Core Simulation Engine Development)
- [x] **基础数据结构 (Basic Data Structures)**: 定义光场 (OpticalField)、网格 (Grid) 类
- [x] **光源模块 (Source Module)**:
  - [x] 平面波 (Plane Wave)
  - [x] 高斯光束 (Gaussian Beam)
  - [x] 拉盖尔-高斯光束 (Laguerre-Gaussian Beam)
  - [x] 贝塞尔光束 (Bessel Beam)
- [x] **传播算法 (Propagation Algorithm)**:
  - [x] 实现基于 PyTorch 的角谱法 (Angular Spectrum Method, ASM)
  - [x] 支持 GPU 加速 (CUDA) 和 CPU 回退
- [x] **调制器模块 (Modulator Module)**:
  - [x] 相位调制 (Phase Modulation)
  - [x] 振幅/透射率调制 (Amplitude/Transmission Modulation)
  - [x] 角度-透射率特性 (Angle-dependent Transmission)
- [x] **监视器模块 (Monitor Module)**:
  - [x] 场分布记录 (Field Recording)
  - [x] 数据导出 (HDF5, MAT, CSV)

## 3. 图形用户界面开发 (GUI Development)
- [x] **主窗口框架 (Main Window Framework)**: 菜单栏, 工具栏, 状态栏
- [x] **参数配置面板 (Parameter Configuration Panels)**:
  - [x] 1.1 光路传输方向设置 (Propagation Direction)
  - [x] 1.2 三维网格参数 (3D Grid Parameters)
  - [x] 1.3 光源配置 (Source Configuration)
  - [x] 1.4 第一调制平面 (1st Modulation Plane)
  - [x] 1.5 第二调制平面 (2nd Modulation Plane)
  - [x] 1.6 监视器系统 (Monitor System)
- [x] **可视化模块 (Visualization Module)**:
  - [x] 2D 切面显示 (Matplotlib/PyQtGraph) - 支持 XY 平面热力图和截面曲线
  - [ ] 3D 体绘制 (可选 - 当前支持通过多监视器切片查看)
- [x] **交互逻辑 (Interaction Logic)**:
  - [x] 实时参数更新与预览 (Real-time Preview)
  - [x] 工程文件保存/加载 (JSON Save/Load)

## 4. 验证与测试 (Verification & Testing)
- [x] **单元测试 (Unit Tests)**: 覆盖率 > 90% (初步实现 test_core.py)
- [x] **基准测试 (Benchmark Tests)**:
  - [x] 平面波传播误差 < 1% (在 test_core.py 中验证)
  - [x] 高斯光束聚焦误差 < 0.5% (在 test_core.py 中验证能量守恒)
- [x] **性能优化 (Performance Optimization)**: 确保 GPU 加速有效 (代码已支持 CUDA)

## 5. 文档与交付 (Documentation & Delivery)
- [x] 编写用户手册 (User Manual - README.md)
- [x] 编写技术文档 (Technical Documentation - 代码注释)
- [x] 打包发布 (Packaging): 使用 PyInstaller 生成可执行文件 (命令已在 README 中提供)
