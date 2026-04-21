import unittest

import numpy as np
import os

from src.gui.file_import_utils import (
    compute_file_grid_validation,
    compute_preview_extent_um,
    remap_imported_matrix_to_sim_grid,
)


class TestFileImportValidation(unittest.TestCase):
    def test_dimension_match(self):
        ok, msg = compute_file_grid_validation(64, 64, [(128, 32), (32, 128)])
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_dimension_mismatch(self):
        ok, msg = compute_file_grid_validation(64, 64, [(63, 64)])
        self.assertFalse(ok)
        self.assertIn("网格总点数与文件矩阵维度不一致", msg)
        self.assertIn("文件：63×64", msg)
        self.assertIn("设定：64×64", msg)

    def test_boundary_positive(self):
        ok, _ = compute_file_grid_validation(1, 1, [(1, 1)])
        self.assertTrue(ok)

    def test_invalid_nx_ny(self):
        ok, msg = compute_file_grid_validation(0, 64, [(64, 64)])
        self.assertFalse(ok)
        self.assertIn("必须为正数", msg)

    def test_ignore_none_shape(self):
        ok, msg = compute_file_grid_validation(16, 16, [None, (32, 8)])
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_preview_extent_formula(self):
        x_min, x_max, y_min, y_max = compute_preview_extent_um(2.0, 3.0, 10, 20, 1.5, -2.0)
        self.assertAlmostEqual(x_min, -8.5, places=9)
        self.assertAlmostEqual(x_max, 11.5, places=9)
        self.assertAlmostEqual(y_min, -32.0, places=9)
        self.assertAlmostEqual(y_max, 28.0, places=9)

    def test_numeric_input_positive_constraint(self):
        # 约束语义测试：非法/非正值应被判定为无效
        candidates = [0, -1, -1e-9]
        for v in candidates:
            ok, _ = compute_file_grid_validation(v, 8, [(4, 16)])
            self.assertFalse(ok)

    def test_remap_padding_shape_and_fill(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        sim_x = np.linspace(-3, 3, 8)
        sim_y = np.linspace(-3, 3, 8)
        out = remap_imported_matrix_to_sim_grid(
            data_2d=src,
            sim_x_um=sim_x,
            sim_y_um=sim_y,
            grid_x_um=1.0,
            grid_y_um=1.0,
            nx=2,
            ny=2,
            center_x_um=0.0,
            center_y_um=0.0,
            fill_value=0.0,
        )
        self.assertEqual(out.shape, (8, 8))
        self.assertTrue(np.isclose(out.max(), 4.0))
        self.assertTrue(np.isclose(out.min(), 0.0))

    def test_monitor_slice_fallback_nonempty(self):
        from src.core.monitor import Monitor
        axis = np.linspace(-2.0, 2.0, 4)  # does not include 0
        m = Monitor(position_z=0.0, name="m", plane_type=0, ranges={'x': (0.0, 0.0)})
        sl, vals = m._get_slice_indices(axis, 'x')
        self.assertEqual(vals.size, 1)
        self.assertEqual((sl.stop - sl.start), 1)

    def test_imported_mask_changes_downstream_field(self):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from src.core.field import Grid
        from src.core.source import PlaneWave
        from src.core.propagator import AngularSpectrumPropagator
        from src.core.modulator import SpatialModulator

        grid = Grid(96, 96, 0.5e-6, 0.5e-6, 532e-9)
        src = PlaneWave(grid, amplitude=1.0).generate(device="cpu")
        propagator = AngularSpectrumPropagator(grid)

        base = propagator.propagate(src, 50e-6)
        i_base = (np.abs(base.Ex.cpu().numpy()) ** 2 + np.abs(base.Ey.cpu().numpy()) ** 2)

        # 导入掩膜（小尺寸）并映射到仿真网格
        raw = np.zeros((12, 12), dtype=np.float32)
        raw[2:10, 2:10] = 1.0
        amp = remap_imported_matrix_to_sim_grid(
            data_2d=raw,
            sim_x_um=grid.X[0, :] * 1e6,
            sim_y_um=grid.Y[:, 0] * 1e6,
            grid_x_um=0.5,
            grid_y_um=0.5,
            nx=12,
            ny=12,
            center_x_um=0.0,
            center_y_um=0.0,
            fill_value=1.0,
        )
        mod = SpatialModulator(grid, amplitude_mask=amp)
        after_mod = mod.modulate(src)
        after_prop = propagator.propagate(after_mod, 50e-6)
        i_mod = (np.abs(after_prop.Ex.cpu().numpy()) ** 2 + np.abs(after_prop.Ey.cpu().numpy()) ** 2)

        diff = np.mean(np.abs(i_mod - i_base))
        self.assertGreater(diff, 1e-8)


if __name__ == "__main__":
    unittest.main()
