#!/usr/bin/env python3
"""
Professional DICOM → 3D Mesh Pipeline
=====================================
Generates a real, printable 3D surface mesh from CT DICOM slices.
No volume rendering, no image stacks — pure geometry extraction.
"""

import os
import numpy as np
import pydicom
import pyvista as pv
from typing import List, Tuple, Dict

# ------------------------------------------------------------------
# 1. Load DICOM volume
# ------------------------------------------------------------------
def load_dicom_volume(directory: str) -> Tuple[np.ndarray, Dict]:
    """Load sorted DICOM slices into a calibrated 3D HU volume."""
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith('.dcm')]
    
    if not files:
        raise FileNotFoundError(f"No .dcm files found in: {directory}")

    # Read metadata & sort by Z position
    slices_meta = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        z = float(getattr(ds, 'SliceLocation', 0))
        if hasattr(ds, 'ImagePositionPatient'):
            z = float(ds.ImagePositionPatient[2])
        slices_meta.append((z, f))

    slices_meta.sort(key=lambda x: x[0])
    n_slices = len(slices_meta)

    # Geometry from first slice
    first = pydicom.dcmread(slices_meta[0][1])
    rows = int(first.Rows)
    cols = int(first.Columns)

    sx = sy = 1.0
    if hasattr(first, 'PixelSpacing') and len(first.PixelSpacing) == 2:
        sx = float(first.PixelSpacing[0])
        sy = float(first.PixelSpacing[1])

    # Z spacing from slice positions
    z_vals = [z for z, _ in slices_meta]
    sz = float(np.mean(np.diff(z_vals))) if n_slices > 1 else 1.0

    # Build 3D HU array: shape (Z, Y, X)
    volume = np.zeros((n_slices, rows, cols), dtype=np.float32)
    for i, (_, filepath) in enumerate(slices_meta):
        ds = pydicom.dcmread(filepath)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        volume[i] = ds.pixel_array.astype(np.float32) * slope + intercept

    meta = {
        'spacing': (sx, sy, sz),      # X, Y, Z in mm
        'shape': volume.shape,        # (Z, Y, X)
        'n_slices': n_slices,
    }
    print(f"[INFO] Volume loaded: {meta['shape']}")
    print(f"[INFO] Spacing: {sx:.3f} x {sy:.3f} x {sz:.3f} mm")
    return volume, meta


# ------------------------------------------------------------------
# 2. Extract surface mesh from HU threshold
# ------------------------------------------------------------------
def extract_surface_mesh(volume: np.ndarray, meta: Dict, hu_threshold: float = 200.0) -> pv.PolyData:
    """
    Extract a real 3D mesh at a given HU isosurface.
    Default 200 HU captures cancellous + cortical bone.
    Use 600+ for dense cortical bone only, or -400 for skin envelope.
    """
    nz, ny, nx = meta['shape']
    sx, sy, sz = meta['spacing']

    # Build PyVista ImageData (structured grid)
    grid = pv.ImageData()
    grid.dimensions = [nx, ny, nz]
    grid.spacing = [sx, sy, sz]
    grid.origin = (0.0, 0.0, 0.0)

    # Flatten in C-order: X varies fastest → matches VTK point ordering
    grid.point_data["HU"] = volume.flatten(order="C")

    print(f"[INFO] Running Marching Cubes at HU = {hu_threshold} ...")
    mesh = grid.contour(isosurfaces=[hu_threshold], scalars="HU")

    if mesh.n_points == 0:
        raise RuntimeError(
            f"No surface generated at HU {hu_threshold}. "
            f"Volume HU range: [{volume.min():.0f}, {volume.max():.0f}]. "
            f"Try a threshold inside this range."
        )
    return mesh


# ------------------------------------------------------------------
# 3. Refine mesh (clean, smooth, decimate, normals)
# ------------------------------------------------------------------
def refine_mesh(mesh: pv.PolyData, target_faces: int = 300_000) -> pv.PolyData:
    """Professional mesh cleanup for 3D printing or real-time rendering."""
    print(f"[INFO] Raw mesh: {mesh.n_points:,} points, {mesh.n_faces:,} faces")

    # 3a. Clean: merge coincident points, remove degenerate cells
    m = mesh.clean()

    # 3b. Smooth: Laplacian with feature preservation (n_iter=80 is typical for CT)
    m = m.smooth(
        n_iter=80,
        feature_angle=45.0,
        boundary_smoothing=False,
        edge_angle=30.0,
        inplace=False
    )

    # 3c. Decimate if too dense (preserves volume, prevents shape collapse)
    if m.n_faces > target_faces:
        reduction = 1.0 - (target_faces / m.n_faces)
        reduction = max(0.1, min(0.9, reduction))
        print(f"[INFO] Decimating by {reduction*100:.0f}% ...")
        m = m.decimate(reduction, volume_preservation=True)

    # 3d. Normals for correct lighting / 3D printing slicers
    m.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        inplace=True
    )

    # 3e. Fill small boundary holes (optional but recommended for printing)
    m.fill_holes(hole_size=10.0, inplace=True)

    print(f"[INFO] Final mesh: {m.n_points:,} points, {m.n_faces:,} faces")
    return m


# ------------------------------------------------------------------
# 4. Main execution
# ------------------------------------------------------------------
def main():
    dicom_dir = r"f:\3D\P001"   # <-- your DICOM folder

    # Load
    volume, meta = load_dicom_volume(dicom_dir)

    # HU Threshold Guide:
    #   -400  → Skin / external soft envelope
    #    150  → All bone (cancellous + cortical)
    #    300  → Dense bone
    #    600  → Cortical bone only
    #   1000  → Very dense bone / calcification
    mesh = extract_surface_mesh(volume, meta, hu_threshold=200.0)

    # Refine
    mesh = refine_mesh(mesh, target_faces=300_000)

    # Export real 3D files
    mesh.save("bone_model.stl")   # 3D printing, Blender, Slicer
    mesh.save("bone_model.obj")   # Rendering, Unity, Unreal
    print("[INFO] Exported: bone_model.stl, bone_model.obj")

    # Interactive local viewer (PyVista native window — no browser)
    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background("black")
    plotter.add_mesh(
        mesh,
        color="ivory",
        specular=0.7,
        specular_power=25,
        smooth_shading=True,
        show_edges=False,
        pickable=True
    )
    plotter.add_axes(line_width=3)
    plotter.camera_position = "iso"
    print("[INFO] Opening 3D viewer...")
    plotter.show()


if __name__ == "__main__":
    main()