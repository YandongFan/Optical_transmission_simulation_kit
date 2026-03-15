import sys
import os
import unittest
from PyQt6.QtWidgets import QApplication
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.main_window import MainWindow

class TestSimulationZRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create App
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def test_z_range_coverage(self):
        """
        Test that YZ monitor with range 0-200um is fully simulated even if modulators end at 10um.
        """
        window = MainWindow()
        
        # Mock Config
        config = {
            'grid': {'nx': 64, 'ny': 64, 'dx': 1.0, 'dy': 1.0, 'wavelength': 0.532},
            'source': {'type_idx': 0, 'amplitude': 1.0, 'z_pos': 0.0, 'normalize': False},
            'mod1': {'z': 10.0, 'type_idx': 0, 'z_unit': 'um'}, # Custom Mask at 10um
            'mod2': {'z': 20.0, 'type_idx': 0, 'z_unit': 'um'}, # At 20um
            'monitors': [
                {
                    'name': 'Monitor YZ',
                    'plane': 1, # YZ
                    'pos': 0.0, # X=0
                    'range1_min': -10, 'range1_max': 10, # Y
                    'range2_min': 0, 'range2_max': 200.0, # Z: 0-200um
                    'output_components': ['Ex']
                }
            ]
        }
        
        # Mock parameter_panel.get_latest_config
        window.parameter_panel.get_latest_config = MagicMock(return_value=config)
        
        # Mock Visualization Panel to avoid plotting overhead (optional, but good for speed)
        # But we need to verify data.
        # MainWindow calls window.visualization_panel.add_monitor_result
        # We can inspect window.visualization_panel.monitor_data
        
        # Run Simulation
        window.on_run()
        
        # Check Results
        mon_data = window.visualization_panel.monitor_data.get('Monitor YZ')
        self.assertIsNotNone(mon_data, "Monitor data should exist")
        
        x_data = mon_data['x'] # For YZ, x is Z-axis (horizontal in plot)
        z_max_actual = x_data.max()
        
        print(f"Simulated Z Max: {z_max_actual:.2f} um")
        
        # Assert
        self.assertGreaterEqual(z_max_actual, 199.0, "Z range should cover up to 200um")
        
        # Verify no crash / empty data
        self.assertIsNotNone(mon_data['intensity'], "Intensity should be present")
        self.assertGreater(mon_data['intensity'].max(), 0.0, "Intensity should not be all zero")

if __name__ == '__main__':
    unittest.main()
