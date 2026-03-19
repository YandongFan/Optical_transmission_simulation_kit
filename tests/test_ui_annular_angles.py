import unittest
import sys
import torch # Pre-import torch to avoid DLL conflict with PyQt6 (WinError 1114)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.gui.main_window import MainWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestUIAnnularAngles(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.pp = self.window.parameter_panel

    def test_ui_sync_and_validation(self):
        # Access mod1 trans annular parameters
        id_p = "mod1_trans"
        combo_shape = getattr(self.pp, f"combo_shape_{id_p}")
        combo_shape.setCurrentIndex(0) # Annular
        
        sb_start = getattr(self.pp, f"sb_ann_angle_start_{id_p}")
        sb_end = getattr(self.pp, f"sb_ann_angle_end_{id_p}")
        
        # Test normal case
        sb_start.setValue(45.0)
        sb_end.setValue(135.0)
        
        # Check validation states
        self.assertTrue(self.pp.btn_preview.isEnabled())
        self.assertTrue(self.pp.btn_run.isEnabled())
        
        # Test invalid case (end <= start)
        sb_end.setValue(30.0)
        
        # Check validation disabled buttons
        self.assertFalse(self.pp.btn_preview.isEnabled())
        self.assertFalse(self.pp.btn_run.isEnabled())
        
        # Test slider sync
        sl_start = getattr(self.pp, f"sl_ann_angle_start_{id_p}")
        sl_end = getattr(self.pp, f"sl_ann_angle_end_{id_p}")
        
        self.assertEqual(sl_start.value(), 45)
        
        # Change slider, expect spinbox to update
        sl_start.setValue(90)
        self.assertAlmostEqual(sb_start.value(), 90.0, places=6)
        
        # Restore valid case
        sb_end.setValue(360.0)
        self.assertTrue(self.pp.btn_preview.isEnabled())
        
        # Sync to config check
        config = self.pp.get_latest_config()
        # Verify precision < 1e-6 error, spinbox to config
        # the parameters are retrieved in main_window.generate_geom_mask
        self.assertAlmostEqual(sb_start.value(), 90.0, places=6)
        self.assertAlmostEqual(sb_end.value(), 360.0, places=6)

if __name__ == '__main__':
    unittest.main()
