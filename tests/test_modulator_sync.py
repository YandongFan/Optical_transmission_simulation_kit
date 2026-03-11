import sys
import os
import unittest
from PyQt6.QtWidgets import QApplication, QWidget

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.parameter_panel import ParameterPanel

# Ensure QApplication exists
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

class TestModulatorSync(unittest.TestCase):
    def setUp(self):
        self.panel = ParameterPanel()
        
    def test_modulator_type_sync(self):
        # Initial state: Custom Mask (0)
        self.assertEqual(self.panel.combo_mod1_type.currentIndex(), 0)
        
        # Change to Ideal Lens (1)
        self.panel.combo_mod1_type.setCurrentIndex(1)
        
        # Check config
        config = self.panel.get_latest_config()
        self.assertEqual(config['mod1']['type_idx'], 1)
        
        # Change to Cylindrical Lens X (2)
        self.panel.combo_mod1_type.setCurrentIndex(2)
        config = self.panel.get_latest_config()
        self.assertEqual(config['mod1']['type_idx'], 2)
        
    def test_modulator_param_sync(self):
        # Set to Ideal Lens to see lens params
        self.panel.combo_mod1_type.setCurrentIndex(1)
        
        # Find the spinboxes
        # They are created dynamically, so we access by name
        # sb_mod1_f
        
        # Change focal length
        self.panel.sb_mod1_f.setValue(200.0) # 200 mm
        
        config = self.panel.get_latest_config()
        # Check 'mod1' -> 'lens' -> 'f'
        # The structure in config depends on get_project_data
        # parameter_panel.get_project_data():
        # mod_data['lens'] = {'f': ..., 'f_unit': ...}
        
        lens_cfg = config['mod1'].get('lens', {})
        self.assertAlmostEqual(lens_cfg.get('f'), 200.0)
        
    def test_modulator_pol_sync(self):
        # Toggle polarization
        # Uncheck Unpolarized, Check Linear X
        self.panel.cb_mod1_pol_unpol.setChecked(False)
        self.panel.cb_mod1_pol_lin_x.setChecked(True)
        
        config = self.panel.get_latest_config()
        pols = config['mod1']['affected_polarizations']
        self.assertNotIn('unpolarized', pols)
        self.assertIn('linear_x', pols)

if __name__ == '__main__':
    unittest.main()
