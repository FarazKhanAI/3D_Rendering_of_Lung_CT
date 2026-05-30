
# COVID-CT-MD 3D Chest CT Scan Visualizer

This project provides an interactive 3D volumetric visualizer for chest CT scans from the COVID-CT-MD dataset. It uses a lightweight Python backend (`visualize.py`) to preprocess DICOM slices, extract real anatomy meshes (lung & bone), and serve the data. The WebGL/Three.js frontend dashboard (`index.html`) renders the 3D volume with interactive controls, clipping planes, and a 2D slice inspector.

---

## 🛠️ Prerequisites & Installation

Since you are already inside your activated Python virtual environment `(3D) PS F:\3D>`, install the dependencies listed in `requirements.txt`. This includes `pydicom` (for reading DICOMs), `numpy` (for grid handling), `pylibjpeg` + `pylibjpeg-libjpeg` (for JPEG Lossless decompression), and `pyvista` (for real 3D mesh extraction via Marching Cubes).

Run the following command:

```powershell
uv pip install -r requirements.txt
```
or:
```powershell
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Start the Backend Server**:
    Execute the `visualize.py` script. It will automatically open your browser and serve the dashboard:
    ```powershell
    python visualize.py
    ```
    This starts a local HTTP server (default `http://127.0.0.1:8000`) and opens the web UI.

2. **Process a Dataset**:
    In the web UI, enter the absolute path to your DICOM folder (e.g., `f:\3D\P001`) and click **Reconstruct 3D Volume**. The backend will:
    - Read and sort all `.dcm` slices
    - Calibrate pixel values to Hounsfield Units (HU)
    - Generate `volume.bin` + `metadata.json`
    - Extract real 3D surface meshes for **Lungs** and **Bone** using Marching Cubes
    - Serve everything to the frontend

3. **View the 3D Dashboard**:
    Once processing completes, the app automatically transitions to the 3D viewer. No manual refresh needed.

---

## 🖥️ Web Dashboard Features (Three.js)

### 3D Viewer
- **Real Anatomy Meshes** — Separate, toggleable 3D surfaces for Lung and Bone extracted from HU thresholds
- **Interactive Rotation** — Full 3D orbit with damping
- **Lock Horizontal Rotation** — Restricts polar angle (up/down tilt) to keep the object upright
- **Lock Vertical Rotation** — Restricts azimuth angle (left/right spin) to keep the object facing forward
- **Reset View** — Instantly re-frames the camera on the anatomy
- **Show Bounds** — Toggle a wireframe bounding box around the volume

### 2D Slice Inspector
- Live axial, sagittal, or coronal slice rendering from the raw `volume.bin`
- **Window/Level** sliders (Min/Max HU) to threshold visibility
- Syncs with the 3D clipping plane position

### 3D Clipping Planes
- **Sagittal, Coronal, Axial** clipping with adjustable plane position
- **Invert Clipping** — Flip which side of the plane is visible
- **Show Plane Guide** — Visual helper indicating the clip boundary

### Volume Metadata Panel
- Dimensions, slice count, pixel spacing, and slice spacing read directly from DICOM headers

---

## 📁 Generated Files

After processing, the following files are created in the working directory:

| File | Description |
|------|-------------|
| `volume.bin` | Normalized 3D volume (uint8) for 2D slice rendering |
| `metadata.json` | Volume dimensions and physical spacing |
| `manifest.json` | List of generated anatomy meshes with colors/visibility |
| `lung.obj` | 3D surface mesh of lung tissue |
| `bone.obj` | 3D surface mesh of bone structures |

---

## 🧠 Architecture

- **Backend (`visualize.py`)**: Python HTTP server with threaded DICOM processing. Uses `pydicom` for I/O, `numpy` for volume stacking, and `pyvista` for mesh extraction.
- **Frontend (`index.html`)**: Vanilla JS + Three.js (r145). Loads `.obj` meshes via `OBJLoader`, renders slices via HTML5 Canvas, and communicates with the backend via REST polling (`/api/status`, `/api/process`).



---
## Author
- Faraz Khan
  
University of Engineering and Applied Sciences, Swat (UEAS Swat)

Date: May-30-2026 
