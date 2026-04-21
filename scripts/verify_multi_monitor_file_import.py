import os
import sys
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.gui.main_window import MainWindow  # noqa: E402


def main():
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "verify_multi_monitor_output.txt"))
    logs = []
    QMessageBox.critical = lambda *a, **k: None
    app = QApplication([])
    w = MainWindow()
    pp = w.parameter_panel

    pp.sb_nx.setValue(2048)
    pp.sb_ny.setValue(2048)
    pp.sb_dx.setValue(0.05)
    pp.sb_dy.setValue(0.05)
    pp.sb_wavelength.setValue(0.532)
    pp.combo_source.setCurrentIndex(0)  # Plane Wave

    pp.combo_mod1_type.setCurrentIndex(0)
    getattr(pp, "mask_tabs_mod1").setCurrentIndex(0)
    pp.load_data("amp1", r"d:\trae program\Optical_transmission_simulation_kit\Meta_MultiRegion_N8_Dout36_Fine50nm_Tx.mat")
    pp.load_data("phase1", r"d:\trae program\Optical_transmission_simulation_kit\Meta_MultiRegion_N8_Dout36_Fine50nm_Phx.mat")
    getattr(pp, "sb_file_nx_mod1").setValue(730)
    getattr(pp, "sb_file_ny_mod1").setValue(730)
    getattr(pp, "sb_file_grid_x_mod1").setValue(0.05)
    getattr(pp, "sb_file_grid_y_mod1").setValue(0.05)
    getattr(pp, "sb_file_center_x_mod1").setValue(0.0)
    getattr(pp, "sb_file_center_y_mod1").setValue(0.0)
    pp.sb_mod1_z.setValue(10.0)
    pp.combo_mod1_z_unit.setCurrentText("um")
    pp.sb_mod2_z.setValue(1000.0)
    pp.combo_mod2_z_unit.setCurrentText("um")

    pp.monitors = []
    for i in range(1, 11):
        z_um = float(i * 10)
        pp.monitors.append({
            "name": f"M{i}",
            "pos": z_um,
            "pos_unit": "um",
            "plane": 0,
            "type": 0,
            "output_components": [],
            "range1_min": -100.0,
            "range1_max": 100.0,
            "range2_min": -100.0,
            "range2_max": 100.0,
        })

    cfg = pp.get_project_data()
    logs.append(f"cfg_mod1_z={cfg['mod1']['z']}{cfg['mod1']['z_unit']}")
    logs.append(f"cfg_file_params={cfg['mod1'].get('file_import_params')}")
    logs.append(f"cfg_monitor_count={len(cfg.get('monitors', []))}")
    logs.append(f"cfg_monitor_positions={[m.get('pos') for m in cfg.get('monitors', [])]}")

    w.on_run()
    md = w.visualization_panel.monitor_data
    logs.append(f"status: {w.status_bar.currentMessage()}")
    logs.append(f"keys: {sorted(md.keys())}")
    nonempty = 0
    for name in sorted(md.keys()):
        arr = md[name].get("intensity")
        if arr is not None and getattr(arr, "size", 0) > 0:
            nonempty += 1
            logs.append(f"{name}: shape={arr.shape}, mean={float(np.nanmean(arr))}, max={float(np.nanmax(arr))}")
        else:
            logs.append(f"{name}: empty")
    logs.append(f"nonempty_count={nonempty}")

    app.quit()
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "verify_multi_monitor_output.txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"ERROR: {repr(e)}")
        raise
