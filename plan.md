# Plan for Project Persistence and Monitor Enhancements

## 1. Project Persistence (Save/Load)
### GUI Changes
- **`src/gui/main_window.py`**:
    - Add a Menu Bar with "File" menu.
    - Add "Save Project" and "Load Project" actions.
    - Implement `save_project()`:
        - Open File Dialog (Save).
        - Collect data from `ParameterPanel`.
        - Write JSON to file.
        - Update Window Title.
    - Implement `load_project()`:
        - Open File Dialog (Open).
        - Read JSON.
        - Update `ParameterPanel`.
        - Trigger `on_preview()` to refresh.

### Logic Changes
- **`src/gui/parameter_panel.py`**:
    - Implement `get_project_data(self) -> dict`: Serialize all inputs (Grid, Source, Mods, Monitors).
    - Implement `load_project_data(self, data: dict)`: Deserialize and populate inputs.
    - Handle versioning (basic check).

## 2. Monitor Settings & Visualization
### Monitor Settings (`src/gui/parameter_panel.py`)
- Update `create_monitor_tab` / `settings_group`:
    - Add `Range 1 Min/Max` and `Range 2 Min/Max` spinboxes.
    - Dynamic labels based on Plane selection (e.g., for XY: X Range, Y Range).
    - Add validation logic (`min < max`).
- Update `monitors` list structure to store these ranges.

### Monitor Logic (`src/core/monitor.py`)
- Update `Monitor` class to accept and store ranges.
- Update `record()` method to respect these ranges (if applicable, or just store metadata for post-processing/slicing).
    - *Decision*: For YZ/XZ planes (which build up slice by slice), we can skip recording if `current_z` is out of Z-range. For the transverse dimension (e.g., X in YZ), we can slice the array.
    - For XY plane, we usually record at specific Z. If Z is fixed, ranges apply to X and Y. We can slice the field.

### Visualization (`src/gui/visualization_panel.py`)
- Update `VisualizationPanel`:
    - Add logic to handle `complex_field` data.
    - Dynamically add/remove tabs for "Real E" and "Imag E".
    - Add "Geometry Preview" (3D) if possible, or interpret "3D Preview" as adding a visual indicator in existing plots (requirements mention "3D Preview Window", I might add a simple 3D bounding box plot using Matplotlib 3D in a new tab).

### Integration (`src/gui/main_window.py`)
- Update `on_run`:
    - Pass range params to `Monitor`.
    - Extract complex data for visualization.
    - Call `visualization_panel` with extra data.

## 3. Testing
- Create a test script `tests/test_new_features.py` to verify:
    - Save/Load consistency.
    - Monitor range enforcement.
    - Complex field data generation.

