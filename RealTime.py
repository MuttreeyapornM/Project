# import cv2
# import numpy as np
# import threading
# import time
# from image_processing import ImageStitcher
# from param_settings import Golf_img_Path

# class ImageProcessor:
#     def __init__(self, camera_ids, car_image_path, display_width=800, display_height=600, map_width=1040, map_height=1191):
#         self.camera_ids = camera_ids
#         self.car = cv2.imread(car_image_path, cv2.IMREAD_UNCHANGED)

#         # Ensure car image is loaded correctly
#         if self.car is None:
#             raise ValueError(f"Error: Cannot load car image from {car_image_path}")

#         self.caps = {cam_id: cv2.VideoCapture(cam_id) for cam_id in camera_ids}

#         # Ensure all cameras are opened correctly
#         for cam_id, cap in self.caps.items():
#             if not cap.isOpened():
#                 print(f"Warning: Camera {cam_id} failed to open.")

#         # Store map dimensions
#         self.map_width = map_width
#         self.map_height = map_height

#         self.car_dst_points = np.float32([
#             [465, 465],
#             [575, 465],
#             [465, 685],
#             [575, 685]
#         ])
        
#         # Load calibration data for each camera
#         self.calibration_data = {
#             cam_id: self.load_calibration_data(cam_id) for cam_id in camera_ids
#         }

#     def load_calibration_data(self, cameraID):
#         yaml_filename = f'yaml/calibration_data_{cameraID}.yaml'
#         fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        
#         if not fs.isOpened():
#             print(f"Error: Unable to open calibration file {yaml_filename}")
#             return None

#         data = {
#             "camera_matrix": fs.getNode("camera_matrix").mat(),
#             "dist_coeffs": fs.getNode("dist_coeffs").mat(),
#             "homography": fs.getNode("homography").mat()
#         }
#         fs.release()
#         return data

#     def process_image(self, image, cameraID):
#         calib = self.calibration_data.get(cameraID, None)
#         if not calib:
#             print(f"Warning: Missing calibration data for camera {cameraID}")
#             return None, None
        
#         camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]
#         img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
#         warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
#         return img_src_undistorted, warped

#     def grab_frame(self, cam_id, cap, images, warped_rgba_, index):
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Warning: Failed to grab frame from camera {cam_id}")
#             return
#         images[index], warped_rgba_[index] = self.process_image(frame, cam_id)

#     def run(self):
#         while True:
#             start_time_total = time.time()
#             images = [None] * len(self.caps)
#             warped_rgba_ = [None] * len(self.caps)
#             threads = []
            
#             for i, (cam_id, cap) in enumerate(self.caps.items()):
#                 thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, i))
#                 threads.append(thread)
#                 thread.start()
            
#             for thread in threads:
#                 thread.join()
            
#             if all(img is not None for img in images):
#                 final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
#                 cv2.imshow("Final Merged Image", final_merged_image)
                
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         for cap in self.caps.values():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     camera_ids = [0, 2, 4, 6]
#     car_image_path = Golf_img_Path
#     processor = ImageProcessor(camera_ids, car_image_path)
#     processor.run()

#-------------------------------------------------------------V2 baddd------------------------
# import cv2
# import os
# import numpy as np
# import time
# import threading
# from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
# from param_settings import img_car , Car_dst_points ,total_w , total_h 

# class ImageProcessor:
#     def __init__(self, video_paths, img_car, display_width=800, display_height=600,  map_width=total_w, map_height=total_h):
#         self.video_paths = video_paths
#         self.car = img_car
#         # กำหนด video capture ตามที่ระบุ
#         self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
#         self.display_width = display_width
#         self.display_height = display_height
#         self.map_width = map_width
#         self.map_height = map_height
#         self.car_dst_points = Car_dst_points
#         self.calibration_data = {
#             cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
#         }

#     def load_calibration_data(self, cameraID):
#         yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
#         fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
#         data = {
#             "camera_matrix": fs.getNode("camera_matrix").mat(),
#             "dist_coeffs": fs.getNode("dist_coeffs").mat(),
#             "homography": fs.getNode("homography").mat()
#         }
#         fs.release()
#         return data

#     def process_image(self, image, cameraID):
#         start_time = time.time()
#         calib = self.calibration_data[cameraID]
#         camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

#         img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
#         if cameraID in ["rear", "right"]:
#             img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

#         warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
#         proc_time = time.time() - start_time
#         return img_src_undistorted, warped, proc_time

#     def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
#         ret, frame = cap.read()
#         if not ret:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             return
#         undistorted, warped, proc_time = self.process_image(frame, cam_id)
#         images[index] = undistorted
#         warped_rgba_[index] = warped
#         processing_times[index] = proc_time

#     def run(self):
#         cam_names = ["Front", "Left", "Rear", "Right"]
#         while True:
#             start_time_total = time.time()
#             images = [None] * len(self.caps)
#             warped_rgba_ = [None] * len(self.caps)
#             processing_times = [0.0] * len(self.caps)
#             threads = []

#             for i, (cam_id, cap) in enumerate(self.caps.items()):
#                 thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()

#             if all(img is not None for img in images):
#                 final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
#                 merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

#                 resized_width = self.display_width // 2
#                 resized_height = self.display_height // 2
#                 resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

#                 for i, img in enumerate(resized_images):
#                     text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
#                     cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 top_row = np.hstack((resized_images[0], resized_images[1]))
#                 bottom_row = np.hstack((resized_images[2], resized_images[3]))
#                 merged_Display_image = np.vstack((top_row, bottom_row))

#                 total_proc_time = (time.time() - start_time_total) * 1000
#                 total_text = f"Total Processing: {total_proc_time:.1f} ms"

#                 resized_final_image = cv2.resize(merged_car_image, (800, 900))
#                 cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 print(total_text)

#                 cv2.imshow("Merged Image", merged_Display_image)
#                 cv2.imshow("Final Merged Image", resized_final_image)
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#         for cap in self.caps.values():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     video_paths = {
#         "front": 0,  # กล้องหน้า (ID 0)
#         "left": 4,   # กล้องซ้าย (ID 4)
#         "rear": 2,   # กล้องหลัง (ID 2)
#         "right": 6,  # กล้องขวา (ID 6)
#     }
#     processor = ImageProcessor(video_paths, img_car)
#     processor.run()

#-----------------------v3 good------------------------------------------------
# import cv2
# import os
# import numpy as np
# import time
# import threading
# from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
# from param_settings import img_car , Car_dst_points ,total_w , total_h 

# class ImageProcessor:
#     def __init__(self, video_paths, img_car, display_width=800, display_height=600,  map_width=total_w, map_height=total_h):
#         self.video_paths = video_paths
#         self.car = img_car
#         # กำหนด video capture ตามที่ระบุและตั้งค่าขนาดภาพเป็น 1280x720
#         self.caps = {key: self.initialize_video_capture(path) for key, path in video_paths.items()}
#         self.display_width = display_width
#         self.display_height = display_height
#         self.map_width = map_width
#         self.map_height = map_height
#         self.car_dst_points = Car_dst_points
#         self.calibration_data = {
#             cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
#         }

#     def initialize_video_capture(self, path):
#         cap = cv2.VideoCapture(path)
#         # ตั้งค่าขนาดภาพเป็น 1280x720
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         return cap

#     def load_calibration_data(self, cameraID):
#         yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
#         fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
#         data = {
#             "camera_matrix": fs.getNode("camera_matrix").mat(),
#             "dist_coeffs": fs.getNode("dist_coeffs").mat(),
#             "homography": fs.getNode("homography").mat()
#         }
#         fs.release()
#         return data

#     def process_image(self, image, cameraID):
#         start_time = time.time()
#         calib = self.calibration_data[cameraID]
#         camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

#         img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
#         if cameraID in ["rear", "right"]:
#             img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

#         warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
#         proc_time = time.time() - start_time
#         return img_src_undistorted, warped, proc_time

#     def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
#         ret, frame = cap.read()
#         if not ret:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             return
#         undistorted, warped, proc_time = self.process_image(frame, cam_id)
#         images[index] = undistorted
#         warped_rgba_[index] = warped
#         processing_times[index] = proc_time

#     def run(self):
#         cam_names = ["Front", "Left", "Rear", "Right"]
#         while True:
#             start_time_total = time.time()
#             images = [None] * len(self.caps)
#             warped_rgba_ = [None] * len(self.caps)
#             processing_times = [0.0] * len(self.caps)
#             threads = []

#             for i, (cam_id, cap) in enumerate(self.caps.items()):
#                 thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()

#             if all(img is not None for img in images):
#                 final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
#                 merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

#                 resized_width = self.display_width // 2
#                 resized_height = self.display_height // 2
#                 resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

#                 for i, img in enumerate(resized_images):
#                     text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
#                     cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 top_row = np.hstack((resized_images[0], resized_images[1]))
#                 bottom_row = np.hstack((resized_images[2], resized_images[3]))
#                 merged_Display_image = np.vstack((top_row, bottom_row))

#                 total_proc_time = (time.time() - start_time_total) * 1000
#                 total_text = f"Total Processing: {total_proc_time:.1f} ms"

#                 resized_final_image = cv2.resize(merged_car_image, (800, 900))
#                 cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 print(total_text)

#                 cv2.imshow("Merged Image", merged_Display_image)
#                 cv2.imshow("Final Merged Image", resized_final_image)
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#         for cap in self.caps.values():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     video_paths = {
#         "front": 0,  # กล้องหน้า (ID 0)
#         "left": 4,   # กล้องซ้าย (ID 4)
#         "rear": 2,   # กล้องหลัง (ID 2)
#         "right": 6,  # กล้องขวา (ID 6)
#     }
#     processor = ImageProcessor(video_paths, img_car)
#     processor.run()

#-------------------v4 10 fps----------------------------------------
# import cv2
# import os
# import numpy as np
# import time
# import threading
# from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
# from param_settings import img_car , Car_dst_points ,total_w , total_h 

# class ImageProcessor:
#     def __init__(self, video_paths, img_car, display_width=800, display_height=600,  map_width=total_w, map_height=total_h):
#         self.video_paths = video_paths
#         self.car = img_car
#         # Initialize video capture with video paths and set the frame size to 1280x720
#         self.caps = {key: self.initialize_video_capture(path) for key, path in video_paths.items()}
#         self.display_width = display_width
#         self.display_height = display_height
#         self.map_width = map_width
#         self.map_height = map_height
#         self.car_dst_points = Car_dst_points
#         self.calibration_data = {
#             cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
#         }

#     def initialize_video_capture(self, path):
#         cap = cv2.VideoCapture(path)
#         # Set video capture frame width and height to 1280x720
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         return cap

#     def load_calibration_data(self, cameraID):
#         yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
#         fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
#         data = {
#             "camera_matrix": fs.getNode("camera_matrix").mat(),
#             "dist_coeffs": fs.getNode("dist_coeffs").mat(),
#             "homography": fs.getNode("homography").mat()
#         }
#         fs.release()
#         return data

#     def process_image(self, image, cameraID):
#         start_time = time.time()
#         calib = self.calibration_data[cameraID]
#         camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

#         img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
#         if cameraID in ["rear", "right"]:
#             img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

#         warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
#         proc_time = time.time() - start_time
#         return img_src_undistorted, warped, proc_time

#     def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
#         ret, frame = cap.read()
#         if not ret:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             return
#         undistorted, warped, proc_time = self.process_image(frame, cam_id)
#         images[index] = undistorted
#         warped_rgba_[index] = warped
#         processing_times[index] = proc_time

#     def run(self):
#         cam_names = ["Front", "Left", "Rear", "Right"]
#         while True:
#             start_time_total = time.time()
#             images = [None] * len(self.caps)
#             warped_rgba_ = [None] * len(self.caps)
#             processing_times = [0.0] * len(self.caps)
#             threads = []

#             for i, (cam_id, cap) in enumerate(self.caps.items()):
#                 thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()

#             if all(img is not None for img in images):
#                 final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
#                 merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

#                 resized_width = self.display_width // 2
#                 resized_height = self.display_height // 2
#                 resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

#                 for i, img in enumerate(resized_images):
#                     text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
#                     cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 top_row = np.hstack((resized_images[0], resized_images[1]))
#                 bottom_row = np.hstack((resized_images[2], resized_images[3]))
#                 merged_Display_image = np.vstack((top_row, bottom_row))

#                 total_proc_time = (time.time() - start_time_total) * 1000
#                 total_text = f"Total Processing: {total_proc_time:.1f} ms"

#                 resized_final_image = cv2.resize(merged_car_image, (800, 900))
#                 cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 print(total_text)

#                 cv2.imshow("Merged Image", merged_Display_image)
#                 cv2.imshow("Final Merged Image", resized_final_image)

#                 # Adding a delay to achieve ~10 FPS
#                 if cv2.waitKey(100) == ord('q'):
#                     break

#         for cap in self.caps.values():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     video_paths = {
#         "front": 0,  # Front camera (ID 0)
#         "left": 4,   # Left camera (ID 4)
#         "rear": 2,   # Rear camera (ID 2)
#         "right": 6,  # Right camera (ID 6)
#     }
#     processor = ImageProcessor(video_paths, img_car)
#     processor.run()

















#-------------------------v5 CUDA processing-------------------
import cv2
import os
import numpy as np
import time
import threading
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        car_img = cv2.imread("golf_car.png", cv2.IMREAD_UNCHANGED)  # Load with alpha if available
        if car_img.shape[2] == 3:  # If no alpha channel, add one
            # Create alpha channel (255 = fully opaque)
            alpha = np.ones((car_img.shape[0], car_img.shape[1], 1), dtype=car_img.dtype) * 255
            car_img = np.concatenate([car_img, alpha], axis=2)
        self.car = car_img
        self.caps = {key: self.initialize_video_capture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
        }

        # Check if CUDA is available
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def initialize_video_capture(self, path):
        cap = cv2.VideoCapture(path)
        # Set the frame size to 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def load_calibration_data(self, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        data = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
            "homography": fs.getNode("homography").mat()
        }
        fs.release()
        return data

    def process_image(self, image, cameraID):
        start_time = time.time()
        calib = self.calibration_data[cameraID]
        camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

        # Use CUDA for undistortion if available
        if self.cuda_available:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            gpu_camera_matrix = cv2.cuda_GpuMat()
            gpu_camera_matrix.upload(camera_matrix)
            gpu_dist_coeffs = cv2.cuda_GpuMat()
            gpu_dist_coeffs.upload(dist_coeffs)
            # Perform GPU-based undistortion
            gpu_undistorted = cv2.cuda.undistort(gpu_image, gpu_camera_matrix, gpu_dist_coeffs)
            img_src_undistorted = gpu_undistorted.download()
        else:
            # Use CPU if CUDA is not available
            img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        if cameraID in ["rear", "right"]:
            img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

        # Use CUDA for perspective warping if available
        if self.cuda_available:
            gpu_undistorted = cv2.cuda_GpuMat()
            gpu_undistorted.upload(img_src_undistorted)
            gpu_homography = cv2.cuda_GpuMat()
            gpu_homography.upload(H)
            # Perform GPU-based perspective warping
            gpu_warped = cv2.cuda.warpPerspective(gpu_undistorted, gpu_homography, (self.map_width, self.map_height))
            warped = gpu_warped.download()
        else:
            # Use CPU if CUDA is not available
            warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))

        proc_time = time.time() - start_time
        return img_src_undistorted, warped, proc_time

    def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        undistorted, warped, proc_time = self.process_image(frame, cam_id)
        images[index] = undistorted
        warped_rgba_[index] = warped
        processing_times[index] = proc_time

    def run(self):
        cam_names = ["Front", "Left", "Rear", "Right"]
        while True:
            start_time_total = time.time()
            images = [None] * len(self.caps)
            warped_rgba_ = [None] * len(self.caps)
            processing_times = [0.0] * len(self.caps)
            threads = []

            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if all(img is not None for img in images):
                final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

                for i, img in enumerate(resized_images):
                    text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
                    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))

                total_proc_time = (time.time() - start_time_total) * 1000
                total_text = f"Total Processing: {total_proc_time:.1f} ms"

                resized_final_image = cv2.resize(merged_car_image, (800, 900))
                cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                print(total_text)

                cv2.imshow("Merged Image", merged_Display_image)
                cv2.imshow("Final Merged Image", resized_final_image)
                if cv2.waitKey(1) == ord('q'):
                    break

        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_paths = {
        "front": 0,  # Front camera (ID 0)
        "left": 4,   # Left camera (ID 4)
        "rear": 2,   # Rear camera (ID 2)
        "right": 6,  # Right camera (ID 6)
    }
    processor = ImageProcessor(video_paths, img_car)
    processor.run()











# import cv2
# import os
# import numpy as np
# import time
# import threading
# from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
# from param_settings import img_car, Car_dst_points, total_w, total_h


# class ImageProcessor:
#     def __init__(self, video_paths, img_car, display_width=800, display_height=600,
#                  map_width=total_w, map_height=total_h, retry_interval=5.0):
#         self.video_paths = video_paths
#         self.cam_names_order = list(video_paths.keys())  # keep original order e.g. front/left/rear/right

#         # ---- Load car overlay image safely ----
#         car_img = cv2.imread("golf_car.png", cv2.IMREAD_UNCHANGED)
#         if car_img is None:
#             print("[WARN] golf_car.png not found or unreadable. Car overlay will be skipped.")
#             self.car = None
#         else:
#             if car_img.shape[2] == 3:  # no alpha channel -> add one
#                 alpha = np.ones((car_img.shape[0], car_img.shape[1], 1), dtype=car_img.dtype) * 255
#                 car_img = np.concatenate([car_img, alpha], axis=2)
#             self.car = car_img

#         self.display_width = display_width
#         self.display_height = display_height
#         self.map_width = map_width
#         self.map_height = map_height
#         self.car_dst_points = Car_dst_points
#         self.retry_interval = retry_interval
#         self._last_retry_time = 0.0

#         # ---- Open cameras, tolerate failures ----
#         # self.caps[cam_id] is either a cv2.VideoCapture (opened) or None (unavailable)
#         self.caps = {}
#         for cam_id, path in video_paths.items():
#             self.caps[cam_id] = self.initialize_video_capture(path)

#         # ---- Load calibration data, tolerate missing/broken yaml ----
#         # self.calibration_data[cam_id] is either a dict of matrices or None
#         self.calibration_data = {}
#         for cam_id in video_paths.keys():
#             self.calibration_data[cam_id] = self.load_calibration_data(cam_id)

#         # A camera is considered "active" only if it both opened AND has valid calibration
#         self._refresh_active_cams()

#         if not self.active_cams:
#             print("[WARN] No cameras are currently active. The system will run with placeholder frames "
#                   "and keep retrying to reconnect.")
#         else:
#             print(f"[INFO] Active cameras: {self.active_cams}")

#         # Pre-built placeholder frames for missing cameras.
#         # NOTE: must match the channel count that real frames produce (3-channel BGR,
#         # since cap.read() -> undistort -> warpPerspective never adds an alpha channel),
#         # otherwise cv2.bitwise_and / stitching will fail with a size/type mismatch.
#         self._placeholder_undistorted = np.zeros((720, 1280, 3), dtype=np.uint8)
#         self._placeholder_warped = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)

#         # Check if CUDA is available
#         try:
#             self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
#         except Exception:
#             self.cuda_available = False

#     # ------------------------------------------------------------------
#     # Setup / capture helpers
#     # ------------------------------------------------------------------
#     def initialize_video_capture(self, path):
#         """Try to open a camera. Returns a VideoCapture if successful, else None."""
#         try:
#             cap = cv2.VideoCapture(path)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#             if not cap.isOpened():
#                 print(f"[WARN] Could not open camera at '{path}'. It will be skipped.")
#                 cap.release()
#                 return None
#             return cap
#         except Exception as e:
#             print(f"[WARN] Exception while opening camera at '{path}': {e}")
#             return None

#     def load_calibration_data(self, cameraID):
#         """Try to load calibration yaml. Returns dict if successful, else None."""
#         yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
#         if not os.path.exists(yaml_filename):
#             print(f"[WARN] Calibration file not found for '{cameraID}': {yaml_filename}")
#             return None
#         try:
#             fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
#             camera_matrix = fs.getNode("camera_matrix").mat()
#             dist_coeffs = fs.getNode("dist_coeffs").mat()
#             homography = fs.getNode("homography").mat()
#             fs.release()
#             if camera_matrix is None or dist_coeffs is None or homography is None:
#                 print(f"[WARN] Calibration data incomplete for '{cameraID}'.")
#                 return None
#             return {
#                 "camera_matrix": camera_matrix,
#                 "dist_coeffs": dist_coeffs,
#                 "homography": homography
#             }
#         except Exception as e:
#             print(f"[WARN] Exception while loading calibration for '{cameraID}': {e}")
#             return None

#     def _refresh_active_cams(self):
#         """A cam is active only if its capture is open AND it has valid calibration data."""
#         self.active_cams = [
#             cam_id for cam_id in self.cam_names_order
#             if self.caps.get(cam_id) is not None and self.calibration_data.get(cam_id) is not None
#         ]

#     def _try_reconnect_inactive_cams(self):
#         """Periodically attempt to reopen cameras / reload calibration that previously failed."""
#         now = time.time()
#         if now - self._last_retry_time < self.retry_interval:
#             return
#         self._last_retry_time = now

#         for cam_id in self.cam_names_order:
#             if self.caps.get(cam_id) is None:
#                 new_cap = self.initialize_video_capture(self.video_paths[cam_id])
#                 if new_cap is not None:
#                     print(f"[INFO] Camera '{cam_id}' reconnected.")
#                     self.caps[cam_id] = new_cap
#             if self.calibration_data.get(cam_id) is None:
#                 new_calib = self.load_calibration_data(cam_id)
#                 if new_calib is not None:
#                     print(f"[INFO] Calibration for '{cam_id}' loaded.")
#                     self.calibration_data[cam_id] = new_calib

#         self._refresh_active_cams()

#     # ------------------------------------------------------------------
#     # Frame processing
#     # ------------------------------------------------------------------
#     def process_image(self, image, cameraID):
#         start_time = time.time()
#         calib = self.calibration_data[cameraID]
#         camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

#         if self.cuda_available:
#             gpu_image = cv2.cuda_GpuMat()
#             gpu_image.upload(image)
#             gpu_camera_matrix = cv2.cuda_GpuMat()
#             gpu_camera_matrix.upload(camera_matrix)
#             gpu_dist_coeffs = cv2.cuda_GpuMat()
#             gpu_dist_coeffs.upload(dist_coeffs)
#             gpu_undistorted = cv2.cuda.undistort(gpu_image, gpu_camera_matrix, gpu_dist_coeffs)
#             img_src_undistorted = gpu_undistorted.download()
#         else:
#             img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

#         if cameraID in ["rear", "right"]:
#             img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

#         if self.cuda_available:
#             gpu_undistorted = cv2.cuda_GpuMat()
#             gpu_undistorted.upload(img_src_undistorted)
#             gpu_homography = cv2.cuda_GpuMat()
#             gpu_homography.upload(H)
#             gpu_warped = cv2.cuda.warpPerspective(gpu_undistorted, gpu_homography, (self.map_width, self.map_height))
#             warped = gpu_warped.download()
#         else:
#             warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))

#         proc_time = time.time() - start_time
#         return img_src_undistorted, warped, proc_time

#     def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
#         """Runs in a thread. Falls back to placeholder frames on any failure."""
#         try:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 # Camera stopped responding mid-run; mark it inactive so run() falls back
#                 # and the reconnect loop will try to bring it back later.
#                 print(f"[WARN] Failed to read frame from '{cam_id}'. Marking camera inactive.")
#                 cap.release()
#                 self.caps[cam_id] = None
#                 images[index] = self._placeholder_undistorted
#                 warped_rgba_[index] = self._placeholder_warped
#                 processing_times[index] = 0.0
#                 return

#             undistorted, warped, proc_time = self.process_image(frame, cam_id)
#             images[index] = undistorted
#             warped_rgba_[index] = warped
#             processing_times[index] = proc_time
#         except Exception as e:
#             print(f"[WARN] Error grabbing/processing frame from '{cam_id}': {e}")
#             images[index] = self._placeholder_undistorted
#             warped_rgba_[index] = self._placeholder_warped
#             processing_times[index] = 0.0

#     # ------------------------------------------------------------------
#     # Main loop
#     # ------------------------------------------------------------------
#     def run(self):
#         cam_names = [name.capitalize() for name in self.cam_names_order]
#         while True:
#             start_time_total = time.time()

#             # Periodically retry cameras/calibration that failed earlier
#             self._try_reconnect_inactive_cams()

#             n = len(self.cam_names_order)
#             images = [self._placeholder_undistorted] * n
#             warped_rgba_ = [self._placeholder_warped] * n
#             processing_times = [0.0] * n

#             threads = []
#             for i, cam_id in enumerate(self.cam_names_order):
#                 if cam_id in self.active_cams:
#                     cap = self.caps[cam_id]
#                     thread = threading.Thread(
#                         target=self.grab_frame,
#                         args=(cam_id, cap, images, warped_rgba_, processing_times, i)
#                     )
#                     threads.append(thread)
#                     thread.start()
#                 # inactive cams simply keep their placeholder entries

#             for thread in threads:
#                 thread.join()

#             # Re-check active list in case grab_frame marked a camera inactive this cycle
#             self._refresh_active_cams()

#             # Proceed as long as we have at least one active camera (or always, using placeholders)
#             final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)

#             if self.car is not None:
#                 merged_car_image = ImageAdjuster.overlay_image_perspective(
#                     final_merged_image.copy(), self.car, self.car_dst_points
#                 )
#             else:
#                 merged_car_image = final_merged_image.copy()

#             resized_width = self.display_width // 2
#             resized_height = self.display_height // 2
#             resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

#             for i, img in enumerate(resized_images):
#                 cam_id = self.cam_names_order[i]
#                 if cam_id in self.active_cams:
#                     text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
#                     color = (0, 255, 0)
#                 else:
#                     text = f"{cam_names[i]}: N/A"
#                     color = (0, 0, 255)
#                 cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # Build display grid robustly for any number of cameras (pads to multiple of 2 if needed)
#             padded_images = resized_images[:]
#             while len(padded_images) % 2 != 0:
#                 padded_images.append(np.zeros((resized_height, resized_width, 3), dtype=np.uint8))

#             rows = []
#             for i in range(0, len(padded_images), 2):
#                 rows.append(np.hstack((padded_images[i], padded_images[i + 1])))
#             merged_Display_image = np.vstack(rows)

#             total_proc_time = (time.time() - start_time_total) * 1000
#             total_text = f"Total Processing: {total_proc_time:.1f} ms | Active: {len(self.active_cams)}/{n}"

#             resized_final_image = cv2.resize(merged_car_image, (800, 900))
#             cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#             print(total_text)

#             cv2.imshow("Merged Image", merged_Display_image)
#             cv2.imshow("Final Merged Image", resized_final_image)
#             if cv2.waitKey(1) == ord('q'):
#                 break

#         for cap in self.caps.values():
#             if cap is not None:
#                 cap.release()
#         cv2.destroyAllWindows()


# if __name__ == '__main__':
#     video_paths = {
#         "front": 0,  # Front camera (ID 0)
#         "left": 2,   # Left camera (ID 4)
#         "rear": 2,   # Rear camera (ID 2)
#         "right": 4,  # Right camera (ID 6)
#     }
#     processor = ImageProcessor(video_paths, img_car)
#     processor.run()


