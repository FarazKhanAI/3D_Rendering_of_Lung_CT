import argparse
import json
import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def load_volume(path='volume.bin', metadata_path='metadata.json'):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    width = int(meta['width'])
    height = int(meta['height'])
    depth = int(meta['depth'])
    spacing = [float(meta.get('spacingX', 1.0)), float(meta.get('spacingY', 1.0)), float(meta.get('spacingZ', 1.0))]

    with open(path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    volume = raw.reshape((depth, height, width))
    volume_xyz = np.transpose(volume, (2, 1, 0)).astype(np.uint8)

    image = vtk.vtkImageData()
    image.SetDimensions(width, height, depth)
    image.SetSpacing(spacing)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().SetScalars(numpy_to_vtk(volume_xyz.ravel(order='C'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    return image, (width, height, depth), spacing


def make_volume_actor(image):
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.00, 0.00, 0.00)    # air / background
    color.AddRGBPoint(25, 0.20, 0.18, 0.16)   # dark tissue
    color.AddRGBPoint(80, 0.95, 0.82, 0.72)   # soft tissue
    color.AddRGBPoint(170, 1.00, 0.98, 0.95)  # bone / dense anatomy

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.00)
    opacity.AddPoint(18, 0.01)
    opacity.AddPoint(45, 0.08)
    opacity.AddPoint(95, 0.25)
    opacity.AddPoint(160, 0.55)
    opacity.AddPoint(255, 0.85)

    property_ = vtk.vtkVolumeProperty()
    property_.SetColor(color)
    property_.SetScalarOpacity(opacity)
    property_.SetInterpolationTypeToLinear()
    property_.ShadeOn()
    property_.SetAmbient(0.25)
    property_.SetDiffuse(0.75)
    property_.SetSpecular(0.25)
    property_.SetSpecularPower(16)
    property_.SetIndependentComponents(True)

    try:
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(image)
    except Exception:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(image)

    mapper.SetBlendModeToComposite()
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(property_)
    return actor


def make_lung_surface_actor(image, lower=25, upper=160):
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdBetween(lower, upper)
    threshold.ReplaceInOn()
    threshold.SetInValue(255)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)
    threshold.Update()

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(threshold.GetOutputPort())
    contour.SetValue(0, 128)
    contour.ComputeNormalsOn()
    contour.ComputeScalarsOff()
    contour.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(contour.GetOutputPort())
    normals.SetFeatureAngle(45)
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.88, 0.82, 0.78)
    actor.GetProperty().SetOpacity(0.60)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.75)
    actor.GetProperty().SetSpecular(0.12)
    actor.GetProperty().SetSpecularPower(12)
    actor.GetProperty().SetInterpolationToPhong()
    return actor


def render(headless=False):
    image, dims, spacing = load_volume()
    width, height, depth = dims

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.03, 0.04, 0.05)

    surface_actor = make_lung_surface_actor(image)
    renderer.AddActor(surface_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1100, 900)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(width * spacing[0] * 1.8, height * spacing[1] * 1.2, depth * spacing[2] * 1.8)
    camera.SetFocalPoint(width * spacing[0] * 0.5, height * spacing[1] * 0.5, depth * spacing[2] * 0.5)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.Zoom(1.15)

    if headless:
        render_window.OffScreenRenderingOn()
        render_window.Render()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(window_to_image.GetOutputPort())
        path = os.path.join(os.getcwd(), 'render_vtk.png')
        writer.SetFileName(path)
        writer.Write()
        print(f'Headless render saved to {path}')
        return

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.GetInteractorStyle().SetDefaultRenderer(renderer)
    render_window.Render()
    interactor.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VTK 3D CT volume renderer')
    parser.add_argument('--headless', action='store_true', help='Render once to an image and exit.')
    args = parser.parse_args()
    render(headless=args.headless)
