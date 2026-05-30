import os
import sys
import json
import webbrowser
import http.server
import socketserver
import threading
import time
import numpy as np
import pydicom

# ------------------------------------------------------------------
# Optional PyVista for real 3D mesh extraction
# ------------------------------------------------------------------
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# ------------------------------------------------------------------
# Thread-safe global state
# ------------------------------------------------------------------
processing_lock = threading.Lock()
processing_state = {
    "status": "idle",
    "progress": 0,
    "current_step": "",
    "logs": [],
    "error": None,
    "metadata": None
}

def log_message(msg):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with processing_lock:
        processing_state["logs"].append(formatted_msg)

def update_state(status=None, progress=None, current_step=None, error=None, metadata=None):
    with processing_lock:
        if status is not None:
            processing_state["status"] = status
        if progress is not None:
            processing_state["progress"] = progress
        if current_step is not None:
            processing_state["current_step"] = current_step
        if error is not None:
            processing_state["error"] = error
        if metadata is not None:
            processing_state["metadata"] = metadata

# ------------------------------------------------------------------
# Anatomy Mesh Generation (Marching Cubes on HU ranges)
# ------------------------------------------------------------------
def generate_anatomy_meshes(volume_hu: np.ndarray, metadata: dict, output_dir: str = "."):
    """
    Extract real 3D surface meshes for Lung and Bone only.
    FAST VERSION: Pre-downsamples lung volume before contouring to avoid huge meshes.
    """
    if not PYVISTA_AVAILABLE:
        log_message("WARNING: PyVista not installed. Skipping 3D mesh generation.")
        return None

    nz, ny, nx = volume_hu.shape
    sx, sy, sz = metadata['spacingX'], metadata['spacingY'], metadata['spacingZ']

    structures = [
        {
            "id": "lung", "name": "Lungs",
            "range": (-1000, -500),
            "color": "#ff6b6b",
            "default": True,
            "downsample": 2,  # Pre-downsample by 2x before extraction
        },
        {
            "id": "bone", "name": "Bone",
            "range": (150, 3000),
            "color": "#f7fff7",
            "default": True,
            "downsample": 1,  # No downsampling
        },
    ]

    manifest = {"meshes": [], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for struct in structures:
        min_hu, max_hu = struct["range"]
        log_message(f"Extracting {struct['name']} surface (HU {min_hu} … {max_hu}) …")

        try:
            # Optional pre-downsampling for lung (reduces data 8x)
            ds = struct.get("downsample", 1)
            if ds > 1:
                log_message(f"  Pre-downsampling {struct['name']} by {ds}x for speed...")
                from scipy.ndimage import zoom
                vol = zoom(volume_hu, (1/ds, 1/ds, 1/ds), order=1)
                dx, dy, dz = sx * ds, sy * ds, sz * ds
            else:
                vol = volume_hu
                dx, dy, dz = sx, sy, sz

            vz, vy, vx = vol.shape

            grid = pv.ImageData()
            grid.dimensions = [vx, vy, vz]
            grid.spacing = [dx, dy, dz]
            grid.origin = (0.0, 0.0, 0.0)

            mask = np.zeros_like(vol, dtype=np.float32)
            mask[(vol >= min_hu) & (vol <= max_hu)] = 1.0
            grid.point_data["mask"] = mask.flatten(order="C")

            surf = grid.contour(isosurfaces=[0.5], scalars="mask")

            if surf.n_points == 0:
                log_message(f"  No geometry found for {struct['name']}.")
                continue

            surf = surf.clean()

            # Light decimation if still dense
            if surf.n_faces > 300_000:
                reduction = 1.0 - (200_000 / surf.n_faces)
                reduction = max(0.3, min(0.8, reduction))
                log_message(f"  Decimating {struct['name']} by {reduction*100:.0f}% …")
                surf = surf.decimate(reduction, volume_preservation=True)

            surf = surf.smooth(n_iter=5, feature_angle=60, boundary_smoothing=False, inplace=False)
            surf.compute_normals(
                point_normals=True, cell_normals=False,
                auto_orient_normals=True, inplace=True
            )

            # Center at origin (use original dimensions for consistency)
            cx = (nx - 1) * sx / 2.0
            cy = (ny - 1) * sy / 2.0
            cz = (nz - 1) * sz / 2.0
            surf.translate([-cx, -cy, -cz], inplace=True)

            out_path = os.path.join(output_dir, f"{struct['id']}.obj")
            surf.save(out_path)

            mb = os.path.getsize(out_path) / (1024 * 1024)
            log_message(f"  Saved {struct['name']}: {surf.n_points:,} pts, {surf.n_faces:,} faces ({mb:.1f} MB)")

            manifest["meshes"].append({
                "id": struct["id"],
                "name": struct["name"],
                "file": f"{struct['id']}.obj",
                "color": struct["color"],
                "defaultVisible": struct["default"],
                "faces": int(surf.n_faces),
                "points": int(surf.n_points)
            })

        except Exception as e:
            log_message(f"  ERROR extracting {struct['name']}: {e}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log_message(f"Anatomy manifest saved: {len(manifest['meshes'])} structure(s).")
    return manifest




# ------------------------------------------------------------------
# Background DICOM Processing
# ------------------------------------------------------------------
def bg_process_dataset(directory):
    try:
        abs_directory = os.path.abspath(directory)
        log_message(f"Starting processing for patient directory: {abs_directory}")
        update_state(progress=2, current_step="Scanning directory for DICOM files")

        if not os.path.isdir(abs_directory):
            raise Exception(f"Directory not found: '{abs_directory}'")

        # 1. Gather DICOM files
        dcm_files = []
        for root, dirs, files in os.walk(abs_directory):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_files.append(os.path.join(root, f))

        if not dcm_files:
            raise Exception(f"No DICOM (.dcm) files found in directory: {abs_directory}")

        log_message(f"Found {len(dcm_files)} DICOM files. Reading slice metadata…")
        update_state(progress=5, current_step="Reading slice metadata")

        # 2. Read metadata & sort
        slices_meta = []
        total_files = len(dcm_files)
        for idx, filepath in enumerate(dcm_files):
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                slice_loc = None
                if hasattr(ds, 'SliceLocation'):
                    slice_loc = float(ds.SliceLocation)
                elif hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
                    slice_loc = float(ds.ImagePositionPatient[2])
                else:
                    base = os.path.basename(filepath)
                    try:
                        num_part = base.split('_')[-1].replace('.dcm', '')
                        slice_loc = float(num_part)
                    except ValueError:
                        slice_loc = 0.0
                slices_meta.append({'filepath': filepath, 'location': slice_loc})
            except Exception as e:
                log_message(f"Warning: Failed to read metadata for {os.path.basename(filepath)}: {e}")

            prog = 5 + int(30 * (idx + 1) / total_files)
            if (idx + 1) % 20 == 0 or (idx + 1) == total_files:
                update_state(progress=prog, current_step=f"Reading metadata: {idx + 1}/{total_files} slices")

        log_message("Sorting slices anatomically …")
        update_state(progress=35, current_step="Sorting slices")
        slices_meta.sort(key=lambda x: x['location'])

        if not slices_meta:
            raise Exception("No readable DICOM slices were found.")

        z_min = slices_meta[0]['location']
        z_max = slices_meta[-1]['location']
        log_message(f"Slice range: Z={z_min:.2f} to Z={z_max:.2f} ({len(slices_meta)} slices)")

        locations = [s['location'] for s in slices_meta]
        if len(locations) > 1:
            diffs = [abs(locations[i] - locations[i-1]) for i in range(1, len(locations))]
            spacing_z = float(np.mean(diffs))
            if spacing_z == 0:
                spacing_z = 1.5
        else:
            spacing_z = 1.5

        # 3. Load pixels
        log_message("Loading pixel arrays and calibrating to Hounsfield Units (HU) …")
        update_state(progress=38, current_step="Calibrating HU intensities")

        first_ds = pydicom.dcmread(slices_meta[0]['filepath'])
        original_rows = int(first_ds.Rows)
        original_cols = int(first_ds.Columns)

        spacing_x, spacing_y = 1.0, 1.0
        if hasattr(first_ds, 'PixelSpacing') and len(first_ds.PixelSpacing) == 2:
            spacing_x = float(first_ds.PixelSpacing[0])
            spacing_y = float(first_ds.PixelSpacing[1])

        downsample_factor = 2
        target_rows = original_rows // downsample_factor
        target_cols = original_cols // downsample_factor

        log_message(f"Original slice size: {original_cols}x{original_rows} px. Grid size: {target_cols}x{target_rows} px.")

        hu_slices = []      # float32 HU volumes for meshing
        volume_slices = []  # uint8 normalized for 2D web inspector
        total_slices = len(slices_meta)

        for idx, s in enumerate(slices_meta):
            try:
                ds = pydicom.dcmread(s['filepath'])
                pixel_array = ds.pixel_array.astype(np.float32)

                slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
                intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
                hu_array = pixel_array * slope + intercept
                hu_downsampled = hu_array[::downsample_factor, ::downsample_factor]

                if hu_downsampled.shape[0] != target_rows or hu_downsampled.shape[1] != target_cols:
                    temp = np.zeros((target_rows, target_cols), dtype=np.float32)
                    r = min(target_rows, hu_downsampled.shape[0])
                    c = min(target_cols, hu_downsampled.shape[1])
                    temp[:r, :c] = hu_downsampled[:r, :c]
                    hu_downsampled = temp

                hu_slices.append(hu_downsampled)

                # Normalize to uint8 for the 2D slice viewer
                min_hu = -1000.0
                max_hu = 1000.0
                normalized = (hu_downsampled - min_hu) / (max_hu - min_hu) * 255.0
                normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
                volume_slices.append(normalized)

            except Exception as e:
                log_message(f"Warning: Failed to process pixel data for slice {idx+1}: {e}")
                hu_slices.append(np.zeros((target_rows, target_cols), dtype=np.float32))
                volume_slices.append(np.zeros((target_rows, target_cols), dtype=np.uint8))

            prog = 38 + int(52 * (idx + 1) / total_slices)
            if (idx + 1) % 20 == 0 or (idx + 1) == total_slices:
                update_state(progress=prog, current_step=f"Processing pixel data: {idx + 1}/{total_slices}")

        # Stack volumes
        volume_hu = np.stack(hu_slices, axis=0)      # (D, H, W) float32
        volume_3d = np.stack(volume_slices, axis=0)  # (D, H, W) uint8

        # 4. Save outputs
        log_message("Reconstruction complete. Saving volume dataset and metadata …")
        update_state(progress=92, current_step="Saving output files")

        volume_path = os.path.join(os.getcwd(), "volume.bin")
        metadata_path = os.path.join(os.getcwd(), "metadata.json")

        with open(volume_path, "wb") as f:
            f.write(volume_3d.tobytes())

        metadata = {
            "width": target_cols,
            "height": target_rows,
            "depth": len(slices_meta),
            "spacingX": spacing_x * downsample_factor,
            "spacingY": spacing_y * downsample_factor,
            "spacingZ": spacing_z
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log_message(f"Volume saved: {metadata['width']}x{metadata['height']}x{metadata['depth']}")

        # 5. Generate real 3D anatomy meshes
        if PYVISTA_AVAILABLE:
            update_state(progress=94, current_step="Extracting 3D anatomy surfaces")
            generate_anatomy_meshes(volume_hu, metadata, output_dir=os.getcwd())
            update_state(progress=98, current_step="Anatomy surfaces ready")
        else:
            log_message("PyVista unavailable — skipping real 3D mesh extraction.")

        update_state(status="completed", progress=100, current_step="Done", metadata=metadata)
        log_message("SUCCESS: All outputs ready (volume.bin + metadata.json + anatomy meshes).")

    except Exception as e:
        error_msg = str(e)
        log_message(f"ERROR: {error_msg}")
        update_state(status="error", error=error_msg)


def start_processing(directory):
    with processing_lock:
        processing_state["status"] = "processing"
        processing_state["progress"] = 0
        processing_state["current_step"] = "Initializing..."
        processing_state["logs"] = []
        processing_state["error"] = None
        processing_state["metadata"] = None

    thread = threading.Thread(target=bg_process_dataset, args=(directory,))
    thread.daemon = True
    thread.start()


# ------------------------------------------------------------------
# HTTP Server
# ------------------------------------------------------------------
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/process':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                directory = data.get('directory')
                if not directory:
                    self.send_json_response(400, {"error": "Directory path is required."})
                    return

                with processing_lock:
                    if processing_state["status"] == "processing":
                        self.send_json_response(400, {"error": "Processing is already in progress."})
                        return

                start_processing(directory)
                self.send_json_response(200, {"status": "started"})
            except Exception as e:
                self.send_json_response(400, {"error": f"Failed to parse request: {str(e)}"})
        else:
            self.send_json_response(404, {"error": "Not Found"})

    def do_GET(self):
        if self.path == '/api/status':
            with processing_lock:
                self.send_json_response(200, processing_state)
        elif self.path == '/api/reset':
            with processing_lock:
                processing_state["status"] = "idle"
                processing_state["progress"] = 0
                processing_state["current_step"] = ""
                processing_state["logs"] = []
                processing_state["error"] = None
                processing_state["metadata"] = None
            self.send_json_response(200, {"status": "reset"})
        else:
            super().do_GET()

    def send_json_response(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, code, message):
        self.send_json_response(code, {"error": message})

def run_server(port=8000):
    try:
        from http.server import ThreadingHTTPServer
        server_class = ThreadingHTTPServer
    except ImportError:
        class ThreadingHTTPServerFallback(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        server_class = ThreadingHTTPServerFallback

    socketserver.TCPServer.allow_reuse_address = True
    with server_class(("127.0.0.1", port), CustomHTTPRequestHandler) as httpd:
        url = f"http://127.0.0.1:{port}"
        print(f"\nServer started at {url}")
        print("Press Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()