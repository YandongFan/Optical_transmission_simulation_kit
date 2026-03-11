
# Tasks

- [ ] **Core Refactoring**
    - [ ] Refactor `OpticalField` in `src/core/field.py` to support `Ex`, `Ey`.
    - [ ] Update `Propagator` in `src/core/propagator.py` to propagate vector fields.
    - [ ] Update `Source` in `src/core/source.py` to generate polarized fields.
    - [ ] Update `Modulator` in `src/core/modulator.py` to support polarization-selective modulation.
    - [ ] Update `Monitor` in `src/core/monitor.py` to record `Ex`, `Ey`, `Ez`.

- [ ] **GUI Updates**
    - [ ] Update `ParameterPanel` in `src/gui/parameter_panel.py` to add polarization controls.
    - [ ] Update `VisualizationPanel` in `src/gui/visualization_panel.py` to support multi-component visualization.
    - [ ] Update `MainWindow` in `src/gui/main_window.py` to link GUI to Core logic.

- [ ] **Testing**
    - [ ] Create unit tests for polarization features in `tests/test_polarization.py`.
    - [ ] Run tests and verify.
