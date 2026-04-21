# 预期坐标轴范围（μm）

用于验证 Preview Plot 坐标映射规则：

- `x_min = center_x - grid_x * Nx / 2`
- `x_max = center_x + grid_x * Nx / 2`
- `y_min = center_y - grid_y * Ny / 2`
- `y_max = center_y + grid_y * Ny / 2`

## Case A (4x4)

- 参数：`grid_x=1.0`, `grid_y=1.0`, `Nx=4`, `Ny=4`, `center_x=1.0`, `center_y=1.0`
- 预期：`x=[-1.0, 3.0]`, `y=[-1.0, 3.0]`

## Case B (3x5)

- 参数：`grid_x=2.0`, `grid_y=1.5`, `Nx=5`, `Ny=3`, `center_x=2.0`, `center_y=3.0`
- 预期：`x=[-3.0, 7.0]`, `y=[0.75, 5.25]`

## Case C (2x6)

- 参数：`grid_x=0.5`, `grid_y=2.0`, `Nx=6`, `Ny=2`, `center_x=1.5`, `center_y=4.0`
- 预期：`x=[0.0, 3.0]`, `y=[2.0, 6.0]`

