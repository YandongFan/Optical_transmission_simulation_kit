import os
import sys
import numpy as np
import h5py

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.core.field import Grid  # noqa: E402
from src.core.source import PlaneWave  # noqa: E402
from src.core.modulator import SpatialModulator  # noqa: E402
from src.core.propagator import AngularSpectrumPropagator  # noqa: E402
from src.gui.file_import_utils import remap_imported_matrix_to_sim_grid  # noqa: E402


def stats(name, arr):
    a = np.asarray(arr)
    finite = float(np.isfinite(a).mean())
    return f"{name}: shape={a.shape}, finite={finite}, min={np.nanmin(a)}, max={np.nanmax(a)}"


def main():
    out = []
    grid = Grid(2048, 2048, 0.05e-6, 0.05e-6, 0.532e-6)
    src = PlaneWave(grid, amplitude=1.0).generate(device="cpu")
    out.append("source ready")

    with h5py.File(r"d:\trae program\Optical_transmission_simulation_kit\Meta_MultiRegion_N8_Dout36_Fine50nm_Tx.mat", "r") as f:
        amp_raw = f["transmittance"][()]
    with h5py.File(r"d:\trae program\Optical_transmission_simulation_kit\Meta_MultiRegion_N8_Dout36_Fine50nm_Phx.mat", "r") as f:
        phase_raw = f["phase"][()]
    out.append(stats("amp_raw", amp_raw))
    out.append(stats("phase_raw", phase_raw))

    sx = grid.X[0, :] * 1e6
    sy = grid.Y[:, 0] * 1e6
    amp = remap_imported_matrix_to_sim_grid(amp_raw, sx, sy, 0.05, 0.05, 730, 730, 0.0, 0.0, 1.0)
    phase = remap_imported_matrix_to_sim_grid(phase_raw, sx, sy, 0.05, 0.05, 730, 730, 0.0, 0.0, 0.0)
    out.append(stats("amp_map", amp))
    out.append(stats("phase_map", phase))

    mod = SpatialModulator(grid, amplitude_mask=amp, phase_mask=phase)
    field = mod.modulate(src)
    i0 = (torch.abs(field.Ex) ** 2 + torch.abs(field.Ey) ** 2).cpu().numpy()
    out.append(stats("I@z=10um(after mod)", i0))

    prop = AngularSpectrumPropagator(grid)
    for k in range(1, 10):
        field = prop.propagate(field, 10e-6)
        Ik = (torch.abs(field.Ex) ** 2 + torch.abs(field.Ey) ** 2).cpu().numpy()
        out.append(stats(f"I@+{(k+1)*10}um", Ik))

    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "debug_file_import_chain_output.txt"))
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
