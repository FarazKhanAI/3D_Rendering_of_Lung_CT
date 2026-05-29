# COVID-CT-MD 3D Chest CT Scan Visualizer

This project provides an interactive 3D volumetric visualizer for Patient `P001`'s chest CT scans from the COVID-CT-MD dataset. It uses a lightweight Python backend to preprocess and serve the data, and a WebGL2/Three.js frontend dashboard to view the volume, rotate it, adjust thresholds, and slice inside using interactive clipping planes.

---

## 🛠️ Prerequisites & Installation

Since you are already inside your activated Python virtual environment `(3D) PS F:\3D>`, you need to install the dependencies listed in `requirements.txt`. This includes `pydicom` (for reading DICOMs), `numpy` (for grid handling), and `pylibjpeg` + `pylibjpeg-libjpeg` (which are required to decompress the JPEG Lossless compressed CT scan slices).

Run the following command to install them:

```powershell
uv pip install -r requirements.txt
```
or:
```powershell
pip install -r requirements.txt
```

---


## 🚀 How to Run

1. **Process the DICOM Data**:
    Execute the `visualize.py` script to generate the 3D volume and metadata:
    ```powershell
    python visualize.py
    ```
    This will create `volume.bin` and `metadata.json` in your workspace.

2. **View Realistic 3D Rendering (VTK)**:
    Run the following command to launch the VTK-based 3D visualizer:
    ```powershell
    python render_vtk.py
    ```
    This will open an interactive 3D window using VTK CPU Ray Casting with realistic transfer functions and lighting. No web dashboard is used for 3D rendering.

---

## 🖥️ 3D Visualization Features (VTK)

* Realistic volume rendering using VTK CPU Ray Casting (no GPU required)
* Anatomical color and opacity transfer functions for soft tissue, muscle, and bone
* Interactive 3D rotation, zoom, and pan
* Lighting and shading for enhanced depth perception

---

> **Note:** The web dashboard and other rendering methods are no longer used for 3D visualization. Only `render_vtk.py` provides the correct, realistic 3D view.
