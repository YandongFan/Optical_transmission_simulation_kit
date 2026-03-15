import unittest
from unittest.mock import MagicMock

class TestZRangeCalculation(unittest.TestCase):
    def test_max_z_calculation(self):
        # 模拟配置数据
        # Scenario: 
        # Modulator at z=10um
        # XY Monitor at z=50um
        # YZ Monitor (Global) with Z range 0-200um
        
        events = [{'z': 10e-6}, {'z': 20e-6}] # Modulators
        xy_monitors = [{'z': 50e-6}] # XY Monitor
        
        # Global Monitor Config (YZ Plane)
        # range2 corresponds to Z axis for YZ plane
        monitors_config = [
            {'plane': 0, 'z': 50.0, 'pos_unit': 'um'}, # XY
            {'plane': 1, 'range2_max': 200.0} # YZ, Z max = 200um
        ]
        
        # Original Logic Simulation
        max_z_original = events[-1]['z']
        if xy_monitors:
            max_xy = max(m['z'] for m in xy_monitors)
            max_z_original = max(max_z_original, max_xy)
            
        print(f"Original Max Z: {max_z_original*1e6} um")
        
        # Expected Max Z should be 200um
        expected_max_z = 200e-6
        
        # Proposed Logic Simulation
        max_z_new = max_z_original
        for m in monitors_config:
            plane_type = m.get('plane', 0)
            if plane_type in [1, 2]: # YZ or XZ
                # range2 is Z
                z_max = m.get('range2_max', -1e9) * 1e-6
                max_z_new = max(max_z_new, z_max)
                
        print(f"New Max Z: {max_z_new*1e6} um")
        
        self.assertAlmostEqual(max_z_new, expected_max_z, delta=1e-9, 
                               msg="Max Z should include global monitor range")

if __name__ == '__main__':
    unittest.main()
