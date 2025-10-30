"""
The main purpose of this file is to capture frames from the camera and send them to the front end using websocket.
FrameCapture object is used to start the camera stream from Orbbec Femto Bolt and get the frames along with the depth data which is saved is .npz format.

Websocket is used to send the frames to the front end.
When a response is recieved from the fr via endpiont trigger is endpoint which is triggered from the front end about the captured frame, streaming thread is stopped and the 'name' of the frame that is recieved in response is saved in the list.
The corresponding .npz file that matches the frame name is also saved in the list.


When other endpoints are triggered which include the metadata about the labels of the frames. The list which includes the dictionary of frame name and the corresponding .npz file gets updated with the new metadata and saved in the DB.

All the data is captures similar to how it is being saved in the sync_and_save.py script.


"""

import argparse
import base64
from json.encoder import py_encode_basestring_ascii
import sys
import os
import time
import logging
import json
from datetime import datetime
import threading
from queue import Queue
import shutil
# from typing_extensions import Required

import cv2
import numpy as np

from pyorbbecsdk import *
from utils import frame_to_bgr_image






class FrameCapture:

    def __init__(self, socketio_connection = None):
        self.logger = self.setup_logging()
        self.camera_status = True

        self.pipeline = Pipeline()
        self.config = Config()

        self.camera_tries = 0

        # while self.camera_tries < 3 and not self.camera_status == True:
        self.camera_tries += 1
        self.camera_initializer = self.camera_setter()

        if not self.camera_status:
            self.logger.error("Failed to initialize camera")
            return
        self.reset_camera_tries()

        # Event used to signal streaming thread to stop (thread-safe)
        self.stop_event = threading.Event()

        # Streaming thread management
        self.streaming_thread = None
        self.streaming_thread_lock = threading.Lock()
        self.is_streaming = False

        # Session management for disk cleanup
        self.current_session_number = 0
        self.frames_in_current_session = 0
        self.max_frames_per_session = 50 #300 -> for final product #NOTE: 50 is for testing
        self.max_sessions = 3
        self.session_lock = threading.Lock()  # Protect session variables

        self.request_queue = Queue()
        self._STOP_SENTINEL = object()  # Unique sentinel for shutdown
        
        # Worker thread management
        self.worker_thread_instance = threading.Thread(target=self.worker_thread, daemon=True)
        self.worker_thread_instance.start()
        self.logger.info("Worker thread started")
        self.socketio_connection = socketio_connection



        
    def reset_camera_tries(self,):
        self.camera_tries = 0
    
    def camera_setter(self):
        """
        Sets the camera profile and starts the pipeline
        """
        try:
            # color
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self.color_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(self.color_profile)
            # depth
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            self.depth_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(self.depth_profile)
            
            self.pipeline.enable_frame_sync()
            self.logger.info("Frame sync enabled")
        except Exception as e:
            self.logger.error(f"Failed to configure streams: {e}")
            self.camera_status = False
        
        try:
            self.pipeline.start(self.config)
            self.logger.info("Pipeline started successfully")
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.camera_status = True
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            self.camera_status = False



    def start_streaming(self,  socketio_connection = None, save_depth_image = False):
        """
        Sends the streaming frames to the front end
        """
        # Frame tracking and logging control
        first_frame = True
        first_socketio_send = True
        frame_count = 0
        log_interval = 100  # Log status every 100 frames
        
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                frames = self.align_filter.process(frames)
                if not frames:
                    continue
                frames  = frames.as_frame_set()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = frame_to_bgr_image(color_frame)

                # Log resolution only on first frame
                if first_frame:
                    color_width = color_frame.get_width()
                    color_height = color_frame.get_height()
                    depth_width = depth_frame.get_width()
                    depth_height = depth_frame.get_height()
                    self.logger.info(f"Stream started - RGB: {color_width}x{color_height}, Depth: {depth_width}x{depth_height}")
                    first_frame = False
                
                # Get dimensions for depth processing
                depth_width = depth_frame.get_width()
                depth_height = depth_frame.get_height()

                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_height, depth_width))
                except ValueError:
                    self.logger.error("Failed to reshape depth data")
                    continue
                # NOTE: this gives you percision in 0.01
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

                # NOTE: this gives you percision in 1mm
                depth_data = depth_data.astype(np.uint16)

                # Session management with thread safety
                with self.session_lock:
                    # Check if current session is full
                    if self.frames_in_current_session >= self.max_frames_per_session:
                        self.logger.info(f"Session {self.current_session_number} is full ({self.max_frames_per_session} frames)")
                        
                        # Check if we need to cleanup oldest session
                        existing_sessions = self._get_existing_sessions('test_RGB')
                        if len(existing_sessions) >= self.max_sessions:
                            self.logger.info(f"Max sessions ({self.max_sessions}) reached, cleaning up oldest session")
                            self._cleanup_oldest_session()
                        
                        # Move to next session
                        self.current_session_number += 1
                        self.frames_in_current_session = 0
                        self.logger.info(f"Starting new session: {self.current_session_number}")
                    
                    # Create session folders if they don't exist
                    self._create_session_folders(self.current_session_number)
                    
                    # Generate frame name with session number
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_name = f"{timestamp}_{self.current_session_number}"
                    
                    # Save to session subfolders
                    rgb_path = os.path.join('test_RGB', str(self.current_session_number), f'{frame_name}.jpg')
                    depth_path = os.path.join('test_depth_data', str(self.current_session_number), f'{frame_name}.npz')
                    
                    cv2.imwrite(rgb_path, color_image)
                    np.savez_compressed(depth_path, depth_data=depth_data)
                    
                    # Increment frame counter
                    self.frames_in_current_session += 1
                
                # Increment global frame counter
                frame_count += 1
                
                # Periodic status logging (every 100 frames at DEBUG level)
                if frame_count % log_interval == 0:
                    self.logger.debug(f"Processed {frame_count} frames (session {self.current_session_number}, {self.frames_in_current_session} in current session)")

                if self.socketio_connection:
                    try:
                        # convert color_image to base64
                        color_image_base64 = base64.b64encode(color_image).decode('utf-8')
                        json_data = json.dumps({"title": "send_frames", "data": {"frame_name": frame_name, "frame_data": color_image_base64}})
                        self.socketio_connection.emit("stream_frames", json_data)
                        
                        # Log only on first successful send
                        if first_socketio_send:
                            self.logger.info(f"First frame sent via WebSocket successfully: {frame_name}")
                            first_socketio_send = False
                        
                    except Exception as e:
                        self.logger.error(f"Failed to send frame_data to front end: {e}")
                else:
                    # Warn only once about missing connection
                    if frame_count == 1:
                        self.logger.warning("Socketio connection not available - frames will not be sent to frontend")


                # NOTE: this is for debugging and testing purpose
                # save the depth image if save_depth_image is True
                if save_depth_image:
                    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                    depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                    if not os.path.exists('test_depth_image'):
                        os.makedirs('test_depth_image')
                    cv2.imwrite(os.path.join('test_depth_image', f'{frame_name}.jpg'), depth_image)
                    # Removed per-frame logging
                # Removed else block - no need to log "not saved"
            except Exception as e:
                self.logger.error(f"Failed to process frames, error: {e}")
                continue
        
        # Summary when streaming stops
        self.logger.info(f"Streaming stopped after processing {frame_count} total frames")

    def _get_existing_sessions(self, base_dir):
        """
        Get list of existing session numbers from a directory
        
        Args:
            base_dir: Directory to check (e.g., 'test_RGB' or 'test_depth_data')
        
        Returns:
            List of session numbers (as integers)
        """
        if not os.path.exists(base_dir):
            return []
        
        sessions = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                sessions.append(int(item))
        
        return sorted(sessions)
    
    def _create_session_folders(self, session_number):
        """
        Create session subfolders in test_RGB and test_depth_data
        
        Args:
            session_number: Session number to create
        """
        rgb_session_dir = os.path.join('test_RGB', str(session_number))
        depth_session_dir = os.path.join('test_depth_data', str(session_number))
        
        if not os.path.exists(rgb_session_dir):
            os.makedirs(rgb_session_dir)
            self.logger.info(f"Created RGB session directory: {rgb_session_dir}")
        
        if not os.path.exists(depth_session_dir):
            os.makedirs(depth_session_dir)
            self.logger.info(f"Created depth session directory: {depth_session_dir}")
    
    def _cleanup_oldest_session(self):
        """
        Delete the oldest (lowest numbered) session folder from both RGB and depth directories
        """
        rgb_sessions = self._get_existing_sessions('test_RGB')
        depth_sessions = self._get_existing_sessions('test_depth_data')
        
        if rgb_sessions:
            oldest_session = min(rgb_sessions)
            rgb_path = os.path.join('test_RGB', str(oldest_session))
            try:
                shutil.rmtree(rgb_path)
                self.logger.info(f"Deleted oldest RGB session: {rgb_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete RGB session {rgb_path}: {e}")
        
        if depth_sessions:
            oldest_session = min(depth_sessions)
            depth_path = os.path.join('test_depth_data', str(oldest_session))
            try:
                shutil.rmtree(depth_path)
                self.logger.info(f"Deleted oldest depth session: {depth_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete depth session {depth_path}: {e}")
    
    def _parse_session_from_filename(self, frame_name):
        """
        Extract session number from frame name
        
        Args:
            frame_name: Frame name in format 'YYYYMMDD_HHMMSS_mmm_<session>'
        
        Returns:
            Session number as integer, or None if parsing fails
        """
        try:
            # Frame name format: 20251024_143025_123_2
            parts = frame_name.split('_')
            if len(parts) >= 4:
                return int(parts[-1])
            return None
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse session from frame name '{frame_name}': {e}")
            return None
    
    def _relocate_selected_frame(self, frame_name, food_level):
        """
        Relocate selected frame and depth data to data_to_process directory
        
        Args:
            frame_name: Name of the selected frame (without extension)
            food_level: Integer food level (must be unique)
        
        Returns:
            bool: True if relocation successful, False otherwise
        """
        # Parse session number
        session_number = self._parse_session_from_filename(frame_name)
        if session_number is None:
            self.logger.error(f"Cannot relocate frame - invalid frame name: {frame_name}")
            return False
        
        # Source paths
        rgb_source = os.path.join('test_RGB', str(session_number), f'{frame_name}.jpg')
        depth_source = os.path.join('test_depth_data', str(session_number), f'{frame_name}.npz')
        
        # Check if files exist
        if not os.path.exists(rgb_source):
            self.logger.error(f"RGB frame not found: {rgb_source}")
            return False
        if not os.path.exists(depth_source):
            self.logger.error(f"Depth data not found: {depth_source}")
            return False
        
        # Destination paths - use food_level instead of level_count
        dest_base = os.path.join('data_to_process', str(food_level))
        
        # Validate food_level is unique (directory shouldn't exist)
        if os.path.exists(dest_base):
            self.logger.error(f"Food level {food_level} already exists - each level must be unique")
            raise ValueError(f"Food level {food_level} directory already exists. Each food level must be unique.")
        
        rgb_dest_dir = os.path.join(dest_base, 'rgb')
        depth_dest_dir = os.path.join(dest_base, 'depth')
        
        # Create destination directories
        os.makedirs(rgb_dest_dir, exist_ok=True)
        os.makedirs(depth_dest_dir, exist_ok=True)
        
        # Copy files
        try:
            shutil.copy2(rgb_source, os.path.join(rgb_dest_dir, f'{frame_name}.jpg'))
            shutil.copy2(depth_source, os.path.join(depth_dest_dir, f'{frame_name}.npz'))
            self.logger.info(f"Relocated frame {frame_name} to food level {food_level}: {dest_base}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to relocate frame {frame_name}: {e}")
            return False
    
    def _delete_all_sessions(self):
        """
        Delete all session folders from test_RGB and test_depth_data
        """
        rgb_sessions = self._get_existing_sessions('test_RGB')
        depth_sessions = self._get_existing_sessions('test_depth_data')
        
        # Delete all RGB sessions
        for session in rgb_sessions:
            rgb_path = os.path.join('test_RGB', str(session))
            try:
                shutil.rmtree(rgb_path)
                self.logger.info(f"Deleted RGB session: {rgb_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete RGB session {rgb_path}: {e}")
        
        # Delete all depth sessions
        for session in depth_sessions:
            depth_path = os.path.join('test_depth_data', str(session))
            try:
                shutil.rmtree(depth_path)
                self.logger.info(f"Deleted depth session: {depth_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete depth session {depth_path}: {e}")
        
        self.logger.info("All session folders deleted")

    def process_request(self, request):
        """
        recieves request and sets flag for worker thread to process the request

        request types can be
        {"type":<str>, "data":<dict>}
        
        type can be:
        "start_streaming"
        "stop_streaming"
        "reset_stream"

        """
        # add to the queue
        self.request_queue.put(request)


    def stop_streaming(self): 
        """
        Stops the streaming frames to the front end, using websocket
        """
        self.stop_event.set()
        print("Streaming stopped")
        
    

    def setup_logging(self):
        """Configure logging with timestamp and proper formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def worker_thread(self):
        """
        Worker thread that processes requests from the queue
        Uses sentinel pattern for efficient shutdown
        """
        self.logger.info("Worker thread running")
        
        while True:  # Infinite loop - exit via sentinel
            try:
                # Block indefinitely until request arrives (most efficient)
                request = self.request_queue.get()
                
                # Check for shutdown sentinel
                if request is self._STOP_SENTINEL:
                    self.logger.info("Shutdown sentinel received")
                    break
                
                self.logger.info(f"Worker received request: {request}")
                
                request_type = request.get("type")
                request_data = request.get("data", {})
                
                if request_type == "start_streaming":
                    self._handle_start_streaming(request_data)
                elif request_type == "stop_streaming":
                    self._handle_stop_streaming(request_data)
                else:
                    self.logger.warning(f"Unknown request type: {request_type}")
                
                self.request_queue.task_done()
                
            except ValueError as ve:
                # Handle validation errors (missing food_level, duplicate)
                self.logger.error(f"Validation error in request: {ve}")
                # Continue processing other requests
                
            except Exception as e:
                # Catch unexpected errors
                self.logger.error(f"Unexpected error in worker thread: {e}", exc_info=True)
        
        self.logger.info("Worker thread exiting")
    
    def _handle_start_streaming(self, data):
        """
        Handles start_streaming request
        Creates streaming thread if not already running
        """
        with self.streaming_thread_lock:
            if self.is_streaming and self.streaming_thread and self.streaming_thread.is_alive():
                self.logger.info("Streaming thread already running, ignoring start request")
                return
            
            # Clear stop event and start streaming
            self.stop_event.clear()
            self.is_streaming = True
            self.streaming_thread = threading.Thread(target=self._streaming_thread_func, daemon=True)
            self.streaming_thread.start()
            self.logger.info("Streaming thread started")
    
    def _handle_stop_streaming(self, data):
        """
        Handles stop_streaming request
        Stops the streaming thread gracefully, relocates selected frame, and cleans up sessions
        
        Args:
            data: Dictionary containing 'frame_name' and 'food_level'
        """
        frame_name = data.get("frame_name", "unknown")
        food_level = data.get("food_level")
        
        # Validate food_level is provided
        if food_level is None:
            self.logger.error("Missing 'food_level' in stop_streaming request - this is required")
            raise ValueError("food_level is required in stop_streaming request")
        
        self.logger.info(f"Stop streaming request received for frame: {frame_name}, food_level: {food_level}")
        
        # Save frame metadata to JSON file
        self._save_selected_frame(frame_name, food_level)
        
        with self.streaming_thread_lock:
            if not self.is_streaming:
                self.logger.info("No streaming thread running, ignoring stop request")
                return
            
            # Signal the streaming thread to stop
            self.stop_event.set()
            self.is_streaming = False
            
        # Wait for streaming thread to finish (with timeout)
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5.0)
            if self.streaming_thread.is_alive():
                self.logger.warning("Streaming thread did not stop within timeout")
            else:
                self.logger.info("Streaming thread stopped successfully")
        
        # Relocate selected frame to data_to_process with food_level
        self.logger.info(f"Relocating selected frame: {frame_name} to food_level: {food_level}")
        try:
            relocation_success = self._relocate_selected_frame(frame_name, food_level)
            
            if relocation_success:
                # Delete all session folders after successful relocation
                self.logger.info("Relocation successful, cleaning up all session folders")
                self._delete_all_sessions()
                
                # Reset session counters for next streaming session
                with self.session_lock:
                    self.current_session_number = 0
                    self.frames_in_current_session = 0
                    self.logger.info("Session counters reset for next streaming session")
            else:
                self.logger.error("Frame relocation failed, keeping session folders for manual recovery")
        except ValueError as e:
            # This catches the duplicate food_level error
            self.logger.error(f"Frame relocation failed: {e}")
            self.logger.error("Keeping session folders for manual recovery")
            raise  # Re-raise to notify caller
    
    def _save_selected_frame(self, frame_name, food_level):
        """
        Saves the selected frame metadata to a JSON file for logging
        
        Args:
            frame_name: Name of the selected frame
            food_level: Integer food level
        """
        try:
            # New location: data_to_process/selected_frames.json
            selected_frames_file = os.path.join('data_to_process', 'selected_frames.json')
            
            # Ensure data_to_process directory exists
            os.makedirs('data_to_process', exist_ok=True)
            
            # Create data structure with all relevant info
            frame_data = {
                "frame_name": frame_name,
                "food_level": food_level,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "captured_at": datetime.now().isoformat(),
                "rgb_file": f"data_to_process/{food_level}/rgb/{frame_name}.jpg",
                "depth_file": f"data_to_process/{food_level}/depth/{frame_name}.npz",
                "status": "pending_relocation"
            }
            
            # Read existing data if file exists
            existing_data = []
            if os.path.exists(selected_frames_file):
                try:
                    with open(selected_frames_file, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]  # Convert old format to list
                except json.JSONDecodeError:
                    self.logger.warning("Existing JSON file corrupted, starting fresh")
                    existing_data = []
            
            # Append new frame data
            existing_data.append(frame_data)
            
            # Write back to JSON file
            with open(selected_frames_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            self.logger.info(f"Saved selected frame metadata to {selected_frames_file}: {frame_name} (food_level: {food_level})")
        except Exception as e:
            self.logger.error(f"Failed to save selected frame metadata: {e}")
    
    def _streaming_thread_func(self):
        """
        Wrapper function for streaming thread
        Calls start_streaming and handles cleanup
        """
        try:
            self.logger.info("Streaming thread function started")
            self.start_streaming()
        except Exception as e:
            self.logger.error(f"Error in streaming thread: {e}")
        finally:
            with self.streaming_thread_lock:
                self.is_streaming = False
            self.logger.info("Streaming thread function exited")
    
    def shutdown(self):
        """
        Gracefully shutdown all threads
        Call this before exiting the application
        """
        self.logger.info("Shutting down FrameCapture...")
        
        # Stop streaming if active
        if self.is_streaming:
            self.stop_event.set()
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=5.0)
        
        # Stop worker thread by sending sentinel (instant wake-up)
        self.request_queue.put(self._STOP_SENTINEL)
        if self.worker_thread_instance and self.worker_thread_instance.is_alive():
            self.worker_thread_instance.join(timeout=1.0)
            if self.worker_thread_instance.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
        
        # Stop pipeline
        try:
            self.pipeline.stop()
            self.logger.info("Pipeline stopped")
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {e}")
        
        self.logger.info("FrameCapture shutdown complete")





### THIS IS FOR TESTING PURPOSES ONLY
if __name__ == "__main__":

    import threading
    import time
    import glob

    def get_random_frame_from_session(session_number=0):
        """Helper to get a random frame name from a session folder"""
        rgb_session_dir = os.path.join('test_RGB', str(session_number))
        if not os.path.exists(rgb_session_dir):
            return None
        
        # Get all jpg files in the session
        jpg_files = glob.glob(os.path.join(rgb_session_dir, '*.jpg'))
        if not jpg_files:
            return None
        
        # Pick a random frame (or middle one)
        selected_file = jpg_files[len(jpg_files) // 2]  # Pick middle frame
        # Extract just the filename without extension
        frame_name = os.path.basename(selected_file).replace('.jpg', '')
        return frame_name

    frame_capture = FrameCapture()

    # Test 1: First streaming session
    print("\n=== Test 1: First Streaming Session ===")
    print("Sending start streaming request...")
    frame_capture.process_request({"type": "start_streaming", "data": {}})
    
    # Let it capture some frames (5 seconds should give plenty)
    print("Capturing frames for 5 seconds...")
    time.sleep(5)
    
    # Get an actual frame name from session 0
    print("Looking for captured frames in session 0...")
    frame_name = get_random_frame_from_session(0)
    
    if frame_name:
        print(f"Found frame: {frame_name}")
        print(f"Sending stop streaming request with frame: {frame_name}, food_level: 1")
        frame_capture.process_request({"type": "stop_streaming", "data": {"frame_name": frame_name, "food_level": 1}})
    else:
        print("ERROR: No frames found! Using fallback name...")
        frame_capture.process_request({"type": "stop_streaming", "data": {"frame_name": "unknown", "food_level": 1}})
    
    # Wait for cleanup to complete
    print("Waiting for cleanup...")
    time.sleep(3)
    
    # Check if data_to_process was created with food_level 1
    if os.path.exists('data_to_process/1/rgb'):
        rgb_files = os.listdir('data_to_process/1/rgb')
        depth_files = os.listdir('data_to_process/1/depth')
        print(f"✓ Success! Relocated {len(rgb_files)} RGB and {len(depth_files)} depth files to food_level 1")
        print(f"  RGB files: {rgb_files}")
        print(f"  Depth files: {depth_files}")
    else:
        print("✗ ERROR: data_to_process/1/ not created!")
    
    # Check selected_frames.json
    if os.path.exists('data_to_process/selected_frames.json'):
        with open('data_to_process/selected_frames.json', 'r') as f:
            metadata = json.load(f)
            print(f"✓ Metadata logged: {len(metadata)} entries")
            print(f"  Latest entry: {metadata[-1]}")
    else:
        print("✗ ERROR: selected_frames.json not created!")
    
    # Check if sessions were cleaned up
    if os.path.exists('test_RGB'):
        remaining_sessions = [d for d in os.listdir('test_RGB') if os.path.isdir(os.path.join('test_RGB', d))]
        if remaining_sessions:
            print(f"✗ WARNING: Session folders still exist: {remaining_sessions}")
        else:
            print("✓ All session folders cleaned up successfully")
    
    # Test 2: Second streaming session
    print("\n=== Test 2: Second Streaming Session ===")
    print("Sending second start streaming request...")
    frame_capture.process_request({"type": "start_streaming", "data": {}})
    
    print("Capturing frames for 3 seconds...")
    time.sleep(3)
    
    # Get an actual frame name from session 0 (restarted)
    print("Looking for captured frames in new session 0...")
    frame_name = get_random_frame_from_session(0)
    
    if frame_name:
        print(f"Found frame: {frame_name}")
        print(f"Sending stop streaming request with frame: {frame_name}, food_level: 2")
        frame_capture.process_request({"type": "stop_streaming", "data": {"frame_name": frame_name, "food_level": 2}})
    else:
        print("ERROR: No frames found!")
        frame_capture.process_request({"type": "stop_streaming", "data": {"frame_name": "unknown", "food_level": 2}})
    
    # Wait for cleanup
    print("Waiting for cleanup...")
    time.sleep(3)
    
    # Check if data_to_process was created with food_level 2
    if os.path.exists('data_to_process/2/rgb'):
        rgb_files = os.listdir('data_to_process/2/rgb')
        depth_files = os.listdir('data_to_process/2/depth')
        print(f"✓ Success! Relocated {len(rgb_files)} RGB and {len(depth_files)} depth files to food_level 2")
        print(f"  RGB files: {rgb_files}")
        print(f"  Depth files: {depth_files}")
    else:
        print("✗ ERROR: data_to_process/2/ not created!")
    
    # Final check: Verify both food levels exist
    print("\n=== Final Verification ===")
    if os.path.exists('data_to_process/selected_frames.json'):
        with open('data_to_process/selected_frames.json', 'r') as f:
            all_metadata = json.load(f)
            print(f"✓ Total frames logged: {len(all_metadata)}")
            for entry in all_metadata:
                print(f"  - Food level {entry['food_level']}: {entry['frame_name']}")
    
    # Verify directory structure
    food_levels = [d for d in os.listdir('data_to_process') if d.isdigit()]
    print(f"✓ Food levels created: {sorted([int(fl) for fl in food_levels])}")
    
    # Shutdown properly
    print("\n=== Shutting Down ===")
    frame_capture.shutdown()
    
    print("\n=== Test Complete ===")










