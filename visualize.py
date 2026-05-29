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

# Thread-safe global processing state
processing_lock = threading.Lock()
processing_state = {
    "status": "idle",       # "idle", "processing", "completed", "error"
    "progress": 0,          # 0 to 100
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

def bg_process_dataset(directory):
    try:
        # Resolve to absolute path
        abs_directory = os.path.abspath(directory)
        log_message(f"Starting processing for patient directory: {abs_directory}")
        update_state(progress=2, current_step="Scanning directory for DICOM files")

        if not os.path.isdir(abs_directory):
            raise Exception(f"Directory not found: '{abs_directory}'")

        # 1. Gather all DICOM files
        dcm_files = []
        for root, dirs, files in os.walk(abs_directory):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_files.append(os.path.join(root, f))
                    
        if not dcm_files:
            raise Exception(f"No DICOM (.dcm) files found in directory: {abs_directory}")
            
        log_message(f"Found {len(dcm_files)} DICOM files. Reading slice metadata...")
        update_state(progress=5, current_step="Reading slice metadata")
        
        # 2. Read and extract slice location/Z-position for sorting
        slices_meta = []
        total_files = len(dcm_files)
        for idx, filepath in enumerate(dcm_files):
            try:
                # Stop before pixel data to read metadata quickly
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                # Slice Location tag is (0020, 1041). Fallback to ImagePositionPatient[2] (Z coordinate)
                slice_loc = None
                if hasattr(ds, 'SliceLocation'):
                    slice_loc = float(ds.SliceLocation)
                elif hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
                    slice_loc = float(ds.ImagePositionPatient[2])
                else:
                    # Fallback to try and extract file sequence number from filename
                    base = os.path.basename(filepath)
                    try:
                        num_part = base.split('_')[-1].replace('.dcm', '')
                        slice_loc = float(num_part)
                    except ValueError:
                        slice_loc = 0.0
                        
                slices_meta.append({
                    'filepath': filepath,
                    'location': slice_loc
                })
            except Exception as e:
                log_message(f"Warning: Failed to read metadata for {os.path.basename(filepath)}: {e}")
            
            # Update progress between 5% and 35%
            prog = 5 + int(30 * (idx + 1) / total_files)
            if (idx + 1) % 20 == 0 or (idx + 1) == total_files:
                update_state(progress=prog, current_step=f"Reading metadata: {idx + 1}/{total_files} slices")
                
        # 3. Sort slices by SliceLocation ascending
        log_message("Sorting slices anatomically to reconstruct correct volume order...")
        update_state(progress=35, current_step="Sorting slices")
        slices_meta.sort(key=lambda x: x['location'])
        
        if not slices_meta:
            raise Exception("No readable DICOM slices were found.")
            
        log_message(f"Slices sorted successfully. Depth range spans from Z={slices_meta[0]['location']:.2f} to Z={slices_meta[-1]['location']:.2f}")
        
        # Calculate spacing Z
        locations = [s['location'] for s in slices_meta]
        if len(locations) > 1:
            diffs = [abs(locations[i] - locations[i-1]) for i in range(1, len(locations))]
            spacing_z = float(np.mean(diffs))
            if spacing_z == 0:
                spacing_z = 1.5 # standard fallback
        else:
            spacing_z = 1.5
            
        # 4. Load slice pixels and construct the 3D volume
        log_message("Loading pixel arrays and calibrating intensity scales to Hounsfield Units (HU)...")
        update_state(progress=38, current_step="Calibrating HU intensities")
        
        # Read first file to get base spacing and size
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
        
        log_message(f"Original slice size: {original_cols}x{original_rows} px. Downsampling grid size: {target_cols}x{target_rows} px.")
        
        volume_slices = []
        total_slices = len(slices_meta)
        
        for idx, s in enumerate(slices_meta):
            try:
                ds = pydicom.dcmread(s['filepath'])
                pixel_array = ds.pixel_array.astype(np.float32)
                
                # Apply Rescale Slope and Intercept for Hounsfield Units (HU)
                slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
                intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
                hu_array = pixel_array * slope + intercept
                
                # Downsample
                hu_downsampled = hu_array[::downsample_factor, ::downsample_factor]
                
                # Pad/crop if shape differs by a pixel due to rounding
                if hu_downsampled.shape[0] != target_rows or hu_downsampled.shape[1] != target_cols:
                    temp = np.zeros((target_rows, target_cols), dtype=np.float32)
                    r = min(target_rows, hu_downsampled.shape[0])
                    c = min(target_cols, hu_downsampled.shape[1])
                    temp[:r, :c] = hu_downsampled[:r, :c]
                    hu_downsampled = temp
                    
                # Normalize HU to uint8 (0-255) in range [-1000, 1000] HU
                min_hu = -1000.0
                max_hu = 1000.0
                normalized = (hu_downsampled - min_hu) / (max_hu - min_hu) * 255.0
                normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
                
                volume_slices.append(normalized)
            except Exception as e:
                log_message(f"Warning: Failed to process pixel data for slice {idx+1}: {e}")
                # Fallback to black slice
                volume_slices.append(np.zeros((target_rows, target_cols), dtype=np.uint8))
                
            # Update progress between 38% and 90%
            prog = 38 + int(52 * (idx + 1) / total_slices)
            if (idx + 1) % 20 == 0 or (idx + 1) == total_slices:
                update_state(progress=prog, current_step=f"Processing pixel data: {idx + 1}/{total_slices}")
                
        # Stack along depth axis to get 3D array: (depth, height, width)
        volume_3d = np.stack(volume_slices, axis=0)
        raw_bytes = volume_3d.tobytes()
        
        # Save output files
        log_message("Reconstruction complete. Saving volume dataset and metadata to workspace...")
        update_state(progress=92, current_step="Saving output files")
        
        volume_path = os.path.join(os.getcwd(), "volume.bin")
        metadata_path = os.path.join(os.getcwd(), "metadata.json")
        
        with open(volume_path, "wb") as f:
            f.write(raw_bytes)
            
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
            
        log_message("Volume binaries package created successfully!")
        log_message(f"Dimensions: {metadata['width']}x{metadata['height']}x{metadata['depth']}. Total voxel bytes: {len(raw_bytes) / (1024*1024):.2f} MB")
        
        update_state(status="completed", progress=100, current_step="Done", metadata=metadata)
        
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

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS and add headers
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
                    self.send_error_response(400, "Directory path is required.")
                    return
                
                with processing_lock:
                    if processing_state["status"] == "processing":
                        self.send_json_response(400, {"status": "error", "error": "Processing is already in progress."})
                        return
                    
                start_processing(directory)
                self.send_json_response(200, {"status": "started"})
            except Exception as e:
                self.send_error_response(400, f"Failed to parse request: {str(e)}")
        else:
            self.send_error_response(404, "Not Found")

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
    # Use ThreadingHTTPServer to handle requests concurrently without blocking
    try:
        from http.server import ThreadingHTTPServer
        server_class = ThreadingHTTPServer
    except ImportError:
        # Fallback for Python < 3.7
        class ThreadingHTTPServerFallback(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        server_class = ThreadingHTTPServerFallback

    socketserver.TCPServer.allow_reuse_address = True
    with server_class(("127.0.0.1", port), CustomHTTPRequestHandler) as httpd:
        url = f"http://127.0.0.1:{port}"
        print(f"\nServer started instantly at {url}")
        print("Press Ctrl+C to stop the server.")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()
