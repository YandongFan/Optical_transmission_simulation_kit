import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.parameter_panel import ParameterPanel

# Use fixture for QApplication
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_unit_conversion(qapp):
    panel = ParameterPanel()
    
    # Use create_unit_spinbox directly
    widget, sb, combo = panel.create_unit_spinbox(1000.0, "um")
    
    # Initial state
    assert combo.currentText() == "um"
    assert sb.value() == 1000.0
    
    # Change to mm
    combo.setCurrentText("mm")
    # 1000 um = 1 mm
    assert sb.value() == 1.0
    
    # Change back to um
    combo.setCurrentText("um")
    assert sb.value() == 1000.0

def test_min_value(qapp):
    panel = ParameterPanel()
    widget, sb, combo = panel.create_unit_spinbox(1.0, "um")
    
    # Set to 0.001
    sb.setValue(0.001)
    assert sb.value() == 0.001
    
    # Check range property
    assert sb.minimum() == 0.001

def test_interdependency(qapp):
    panel = ParameterPanel()
    
    # Access mod1_D and mod1_f
    sb_D = panel.sb_mod1_D
    combo_D = panel.combo_mod1_D_unit
    sb_f = panel.sb_mod1_f
    combo_f = panel.combo_mod1_f_unit
    sb_NA = panel.sb_mod1_NA
    
    # Set D=10mm, f=100mm -> NA = 10 / 200 = 0.05
    combo_D.setCurrentText("mm")
    sb_D.setValue(10.0)
    
    combo_f.setCurrentText("mm")
    sb_f.setValue(100.0)
    
    # NA should update
    assert abs(sb_NA.value() - 0.05) < 1e-4
    
    # Change f unit to um -> 100000 um. NA should stay same.
    combo_f.setCurrentText("um")
    assert sb_f.value() == 100000.0
    assert abs(sb_NA.value() - 0.05) < 1e-4
