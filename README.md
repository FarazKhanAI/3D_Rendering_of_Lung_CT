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

1. **Start the Data Processor & Server**:
   Execute the `visualize.py` script:
   ```powershell
   python visualize.py
   ```

2. **What this command does**:
   - Finds the DICOM scan slices inside the `P001` folder.
   - Reads slice data and **correctly sorts them** using the `(0020, 1041) Slice Location` tag to ensure anatomical ordering.
   - Converts raw pixel data into physical **Hounsfield Units (HU)**.
   - Downsamples slices (to `256x256` for performance) and builds a unified 3D volume, saving it as `volume.bin` (~9.7 MB).
   - Extracts voxel dimensions (pixel size & slice thickness) and saves them as `metadata.json` to preserve correct physical proportions in 3D.
   - Starts a lightweight local web server on port `8000`.
   - **Automatically opens** your default web browser to the viewer dashboard at `http://127.0.0.1:8000`.

---

## 🖥️ Interactive Web Dashboard Features

Once the page loads, you can use the following controls:

*   **Original CT Grayscale Rendering**: Dense structures (like bones and consolidations) are rendered white, while low-density areas (like air inside lungs) are rendered black/transparent, matching clinical PACS workstations.
*   **Horizontal-only Rotation**: Enabled by default (controlled via the **"Lock Horizontal Rotation"** toggle). This locks vertical movement so you can drag left/right to rotate the patient's body horizontally, making analysis stable and easy to control.
*   **Anatomical Clipping Planes**:
    - Choose an axis: **Sagittal** (Left-to-Right), **Coronal** (Front-to-Back), or **Axial** (Head-to-Toe).
    - Drag the **Plane Position** slider to slice through the 3D volume, exposing internal structures.
    - Check **Invert Clipping** to slice in the opposite direction.
    - An overlay plane guide (red outline) displays in 3D to show exactly where you are cutting.
*   **2D Slice Inspector**: A secondary screen in the control panel displays the corresponding **2D slice** (cross-section) of the current clipping plane in real time, with the same color/threshold settings.
*   **Windowing & Threshold Sliders**:
    - **Min HU Cutoff**: Filters out voxels below the selected Hounsfield Unit. Setting this around `-700 HU` removes air and shows lung tissue. Setting this to `+100 HU` or higher isolates the skeletal structure (bones).
    - **Opacity & Contrast**: Fine-tune the transparency and gamma brightness to enhance specific details or highlight potential infection areas.
*   **Visual Modes**: Toggle between **Volume Rendering** (standard raymarching), **MIP (Maximum Intensity Projection)**, and **Isosurface** (solid shell at the threshold value).
*   **Alternative Color Maps**: Easily switch between standard Grayscale, Hot Iron (thermal), Cool Blue, and a multi-color Anatomical view.
