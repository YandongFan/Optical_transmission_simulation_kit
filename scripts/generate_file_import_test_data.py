import os
import numpy as np
from h5py import File


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "file_import")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("case_a_64x64", (64, 64)),
        ("case_b_128x32", (128, 32)),
        ("case_c_45x91", (45, 91)),
    ]

    for name, shape in cases:
        h, w = shape
        y = np.linspace(-1.0, 1.0, h)[:, None]
        x = np.linspace(-1.0, 1.0, w)[None, :]
        phase = np.arctan2(y, x).astype(np.float64)
        trans = np.clip(np.exp(-(x**2 + y**2) * 3.0), 0.0, 1.0).astype(np.float64)

        # 采用 MATLAB v7.3(HDF5) 兼容写法，便于当前环境生成
        with File(os.path.join(out_dir, f"{name}_phase.mat"), "w") as f:
            f.create_dataset("phase_map", data=phase)
        with File(os.path.join(out_dir, f"{name}_trans.mat"), "w") as f:
            f.create_dataset("trans_map", data=trans)

    print(f"Generated test data in: {out_dir}")


if __name__ == "__main__":
    main()
