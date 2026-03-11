import unittest
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.parameter_panel import ParameterPanel

class TestSourceSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure QApplication exists for widgets
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.panel = ParameterPanel()
        
    def test_initial_sync(self):
        """Test that config is initialized on startup"""
        config = self.panel.get_latest_config()
        self.assertIn('grid', config)
        self.assertIn('source', config)
        # Check default values
        self.assertEqual(config['grid']['nx'], 512)
        self.assertEqual(config['source']['amplitude'], 1.0)

    def test_sync_on_change(self):
        """Test that changing UI elements updates the config immediately"""
        
        # 1. Change Wavelength (DoubleSpinBox)
        self.panel.sb_wavelength.setValue(0.633)
        config = self.panel.get_latest_config()
        self.assertAlmostEqual(config['grid']['wavelength'], 0.633)
        
        # 2. Change Source Type (ComboBox)
        self.panel.combo_source.setCurrentIndex(1) # Gaussian
        config = self.panel.get_latest_config()
        self.assertEqual(config['source']['type_idx'], 1)
        
        # 3. Change Polarization Type (ComboBox)
        self.panel.combo_pol_type.setCurrentIndex(1) # LCP
        config = self.panel.get_latest_config()
        self.assertEqual(config['source']['polarization_type'], 1)
        
        # 4. Change Polarization Angle (DoubleSpinBox)
        self.panel.sb_pol_angle.setValue(45.0)
        config = self.panel.get_latest_config()
        self.assertEqual(config['source']['linear_angle'], 45.0)
        
        # 5. Change Custom Equation (TextEdit)
        self.panel.txt_equation.setPlainText("np.exp(-x**2)")
        # textChanged signal is connected
        config = self.panel.get_latest_config()
        self.assertEqual(config['source']['custom']['equation'], "np.exp(-x**2)")
        
        # 6. Change Amplitude (DoubleSpinBox)
        self.panel.sb_amplitude.setValue(5.0)
        config = self.panel.get_latest_config()
        self.assertEqual(config['source']['amplitude'], 5.0)

    def test_sync_mutex_thread_safety(self):
        """Test reading config from background thread while UI updates in main thread"""
        import threading
        import time
        
        stop_event = threading.Event()
        
        def read_config_loop():
            while not stop_event.is_set():
                cfg = self.panel.get_latest_config()
                # Access some data to ensure dict is valid
                _ = cfg['source']['amplitude']
                # time.sleep(0.001)
                
        t_read = threading.Thread(target=read_config_loop)
        t_read.start()
        
        try:
            # Main thread updates UI
            for i in range(100):
                self.panel.sb_amplitude.setValue(float(i))
                QApplication.processEvents() # Allow signals to process
        finally:
            stop_event.set()
            t_read.join()
        
        # Final check
        cfg = self.panel.get_latest_config()
        self.assertEqual(cfg['source']['amplitude'], 99.0)

if __name__ == '__main__':
    unittest.main()
