import numpy as np


def compute_file_grid_validation(nx, ny, file_shapes):
    """
    异步校验核心逻辑（纯函数，便于单元测试）
    file_shapes: [(rows, cols), ...]
    """
    try:
        nx_i = int(nx)
        ny_i = int(ny)
    except Exception:
        return False, "参数无效：Nx / Ny 不是整数"

    if nx_i <= 0 or ny_i <= 0:
        return False, "参数无效：Nx / Ny 必须为正数"

    expected_total = nx_i * ny_i
    for shp in file_shapes:
        if shp is None:
            continue
        rows, cols = int(shp[0]), int(shp[1])
        if rows <= 0 or cols <= 0:
            return False, "文件矩阵维度非法"
        if rows * cols != expected_total:
            return False, (
                f"网格总点数与文件矩阵维度不一致（文件：{rows}×{cols}，设定：{nx_i}×{ny_i}）"
            )

    return True, ""


def compute_preview_extent_um(grid_x_um, grid_y_um, nx, ny, center_x_um, center_y_um):
    gx = float(grid_x_um)
    gy = float(grid_y_um)
    nx_i = int(nx)
    ny_i = int(ny)
    cx = float(center_x_um)
    cy = float(center_y_um)
    x_min = cx - gx * nx_i / 2.0
    x_max = cx + gx * nx_i / 2.0
    y_min = cy - gy * ny_i / 2.0
    y_max = cy + gy * ny_i / 2.0
    return x_min, x_max, y_min, y_max


def remap_imported_matrix_to_sim_grid(
    data_2d,
    sim_x_um,
    sim_y_um,
    grid_x_um,
    grid_y_um,
    nx,
    ny,
    center_x_um,
    center_y_um,
    fill_value,
):
    """
    将导入矩阵映射/补全到仿真网格：
    - 在导入矩阵覆盖范围内，按最近邻映射到仿真网格；
    - 覆盖范围外使用 fill_value 补全。
    """
    arr = np.asarray(data_2d)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"导入矩阵必须为二维，当前为: {arr.shape}")

    h, w = arr.shape
    if h <= 0 or w <= 0:
        raise ValueError("导入矩阵为空")

    # 防止导入文件中 NaN/Inf 在传播中经 FFT 扩散为整幅 NaN
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

    x_min, x_max, y_min, y_max = compute_preview_extent_um(
        grid_x_um, grid_y_um, nx, ny, center_x_um, center_y_um
    )

    # 使用导入矩阵像素数构建采样坐标，保证原始数据不被改写，仅做映射和边界补全
    x_in = np.linspace(x_min, x_max, w, dtype=np.float64)
    y_in = np.linspace(y_min, y_max, h, dtype=np.float64)

    sim_x = np.asarray(sim_x_um, dtype=np.float64)
    sim_y = np.asarray(sim_y_um, dtype=np.float64)

    out = np.full((sim_y.size, sim_x.size), fill_value, dtype=arr.dtype)

    if w == 1:
        ix = np.zeros(sim_x.shape, dtype=np.int64)
        valid_x = np.abs(sim_x - x_in[0]) < 1e-12
    else:
        dx_in = x_in[1] - x_in[0]
        ix = np.rint((sim_x - x_in[0]) / dx_in).astype(np.int64)
        valid_x = (ix >= 0) & (ix < w)

    if h == 1:
        iy = np.zeros(sim_y.shape, dtype=np.int64)
        valid_y = np.abs(sim_y - y_in[0]) < 1e-12
    else:
        dy_in = y_in[1] - y_in[0]
        iy = np.rint((sim_y - y_in[0]) / dy_in).astype(np.int64)
        valid_y = (iy >= 0) & (iy < h)

    if not np.any(valid_x) or not np.any(valid_y):
        return out

    out_x = np.where(valid_x)[0]
    out_y = np.where(valid_y)[0]
    in_x = ix[valid_x]
    in_y = iy[valid_y]

    out[np.ix_(out_y, out_x)] = arr[np.ix_(in_y, in_x)]
    return out
