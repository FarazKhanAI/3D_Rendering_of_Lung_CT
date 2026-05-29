import vtk
import numpy as np
import json

# Load metadata
with open('metadata.json', 'r') as f:
    meta = json.load(f)

width = meta['width']
height = meta['height']
depth = meta['depth']
spacing = [meta['spacingX'], meta['spacingY'], meta['spacingZ']]

# Load volume data
with open('volume.bin', 'rb') as f:
    volume = np.frombuffer(f.read(), dtype=np.uint8)
volume = volume.reshape((depth, height, width))

# Convert numpy array to VTK image
vtk_data = vtk.vtkImageData()
vtk_data.SetDimensions(width, height, depth)
vtk_data.SetSpacing(spacing)
vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

flat = volume.flatten(order='C')
vtk_array = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
vtk_data.GetPointData().SetScalars(vtk_array)

# Set up transfer functions (window: -1000 to 1000 HU mapped to 0-255)
color_func = vtk.vtkColorTransferFunction()
color_func.AddRGBPoint(0, 0.8, 0.4, 0.4)    # Soft tissue (pinkish)
color_func.AddRGBPoint(40, 0.9, 0.7, 0.6)   # Muscle
color_func.AddRGBPoint(80, 0.95, 0.85, 0.7) # Fat
color_func.AddRGBPoint(120, 1.0, 1.0, 1.0)  # Bone
color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)  # Max

opacity_func = vtk.vtkPiecewiseFunction()
opacity_func.AddPoint(0, 0.00)    # Air
opacity_func.AddPoint(30, 0.01)   # Soft tissue
opacity_func.AddPoint(80, 0.15)   # Muscle
opacity_func.AddPoint(120, 0.5)   # Bone
opacity_func.AddPoint(255, 0.8)   # Max

# Volume property
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_func)
volume_property.SetScalarOpacity(opacity_func)
volume_property.ShadeOn()
volume_property.SetInterpolationTypeToLinear()
volume_property.SetAmbient(0.3)
volume_property.SetDiffuse(0.7)
volume_property.SetSpecular(0.2)

# Ray cast mapper (CPU)
mapper = vtk.vtkFixedPointVolumeRayCastMapper()
mapper.SetInputData(vtk_data)

# Volume actor
volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(mapper)
volume_actor.SetProperty(volume_property)

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume_actor)
renderer.SetBackground(0.1, 0.1, 0.1)

# Render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)

# Interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Add SSAO-like effect (approximate with light)
light = vtk.vtkLight()
light.SetLightTypeToSceneLight()
light.SetPosition(0, 0, 1)
light.SetFocalPoint(0, 0, 0)
light.SetIntensity(0.8)
renderer.AddLight(light)

# Start rendering
render_window.Render()
interactor.Start()
