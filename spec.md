
# Specification: Polarization and Vector Field Extension

## 1. User Stories
1.  **Source Polarization**: As a user, I want to select the polarization state of the light source (Linear X, LCP, RCP, Unpolarized) so that I can simulate polarized light propagation.
2.  **Source Preview**: As a user, I want to see the Ex and Ey components of the source field in the preview window to verify the polarization state.
3.  **Modulator Polarization Sensitivity**: As a user, I want to configure modulators (Plane1/Plane2) to affect only specific polarization states (Linear X, LCP, RCP) so that I can simulate polarization-dependent devices.
4.  **Monitor Vector Components**: As a user, I want to choose which field components (Ex, Ey, Ez) to monitor and visualize them separately to analyze the vector nature of the field.

## 2. Architecture Changes

### 2.1 Core
-   **`src/core/field.py`**:
    -   Refactor `OpticalField` to store `Ex` and `Ey` tensors instead of a single `E` tensor.
    -   Add `Ez` calculation method (optional, for monitor output).
    -   Update `get_intensity` to return `|Ex|^2 + |Ey|^2`.
    -   Update `normalize` to normalize based on total intensity.
-   **`src/core/source.py`**:
    -   Update `Source.generate` to return `OpticalField` with initialized `Ex` and `Ey`.
    -   Implement polarization logic:
        -   Linear X (angle $\theta$): $E_x = A \cos\theta, E_y = A \sin\theta$
        -   LCP: $E_x = A/\sqrt{2}, E_y = i A/\sqrt{2}$ (Ex leads Ey by 90 deg? No, usually defined as phase diff. LCP: $\delta = \phi_y - \phi_x = \pi/2$? Need to check user definition: "Ex leads Ey 90 deg" -> $\phi_x = \phi_y + \pi/2$ -> $E_x = E_0 e^{i\pi/2}, E_y = E_0$. Wait, user said "Ex leads Ey 90". So Ex has +90 phase relative to Ey. $E_x = i, E_y = 1$.
        -   RCP: "Ey leads Ex 90" -> $E_y = i, E_x = 1$.
        -   Unpolarized: Monte Carlo or random phase/amplitude between Ex/Ey per pixel/run? User said "Monte Carlo model".
-   **`src/core/propagator.py`**:
    -   Update `AngularSpectrumPropagator.propagate` to propagate `Ex` and `Ey` independently using the same transfer function (isotropic medium).
-   **`src/core/modulator.py`**:
    -   Update `Modulator.modulate` to handle vector fields.
    -   Implement Jones matrix application for polarization-sensitive modulation.
    -   Add `affected_polarizations` logic.
-   **`src/core/monitor.py`**:
    -   Update `record` to store `Ex`, `Ey`, and optionally `Ez`.
    -   Update storage methods (`save_hdf5`, etc.) to save vector components.

### 2.2 GUI
-   **`src/gui/parameter_panel.py`**:
    -   Add "Polarization Type" dropdown to Source tab.
    -   Add "Affected Polarization" checkboxes to Modulator tabs.
    -   Add "Extra Components" checkboxes to Monitor tab.
    -   Update `get_project_data` and `load_project_data` to handle new parameters.
-   **`src/gui/visualization_panel.py`**:
    -   Add support for displaying multiple tabs/plots for a single monitor result (Intensity, Phase, Ex, Ey, Ez).
-   **`src/gui/main_window.py`**:
    -   Update `on_preview` to generate and display Ex/Ey.
    -   Update `on_run` to configure source/modulators/monitors with new parameters.

## 3. Data Structure Updates
-   **Project JSON**:
    -   Source: `polarization_type` (int/str), `linear_angle` (float), `phase_offset` (float).
    -   Modulator: `affected_polarizations` (list of flags).
    -   Monitor: `output_components` (list of flags: 'Ex', 'Ey', 'Ez').

## 4. Testing
-   Unit tests for vector field initialization and normalization.
-   Unit tests for source polarization generation.
-   Unit tests for modulator Jones matrix application.
-   Integration test for full simulation run with vector fields.
