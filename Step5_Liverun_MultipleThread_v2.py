import cv2
import glob
import os
import numpy as np
import time
import threading
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car , Car_dst_points ,total_w , total_h 

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600,  map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        self.car = img_car
        self.camera_order = ["front", "left", "rear", "right"]
        self.image_files = {}
        self.caps = {}
        self.frame_index = 0
        for key, path in video_paths.items():
            files = self._resolve_image_files(path)
            if files:
                self.image_files[key] = files
                print(f"{key}: loaded {len(files)} image files from {path}")
            else:
                self.caps[key] = cv2.VideoCapture(path)
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in ["front", "left", "rear", "right"]
        }

    def _resolve_image_files(self, path):
        if os.path.isdir(path):
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(path, pattern)))
            return sorted(files)
        if any(ch in path for ch in "*?[]"):
            return sorted(glob.glob(path))
        return []

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

        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        if cameraID in ["rear", "right"]:
            img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        proc_time = time.time() - start_time
        return img_src_undistorted, warped, proc_time

    def grab_frame(self, cam_id, images, warped_rgba_, processing_times, index):
        if cam_id in self.image_files:
            files = self.image_files[cam_id]
            frame_path = files[self.frame_index % len(files)]
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Could not read image: {frame_path}")
                return
        else:
            cap = self.caps[cam_id]
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
            images = [None] * len(self.camera_order)
            warped_rgba_ = [None] * len(self.camera_order)
            processing_times = [0.0] * len(self.camera_order)
            threads = []

            for i, cam_id in enumerate(self.camera_order):
                thread = threading.Thread(target=self.grab_frame, args=(cam_id, images, warped_rgba_, processing_times, i))
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
                self.frame_index += 1

        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # video_paths = {
    #     "front": "../Dataset/liverun_outdoor_Day/front.mp4",
    #     "left": "../Dataset/liverun_outdoor_Day/left.mp4",
    #     "rear": "../Dataset/liverun_outdoor_Day/rear.mp4",
    #     "right": "../Dataset/liverun_outdoor_Day/right.mp4",
    # }

# สำหรับโฟลเดอร์รูปภาพ Dataset/cameras
    video_paths = {
    #   "front": "Dataset/camerasN/front",
    #   "left": "Dataset/camerasN/left",
    #   "rear": "Dataset/camerasN/rear",
    #   "right": "Dataset/camerasN/right",
      "front": "Dataset/liverun_indoor/front",
      "left": "Dataset/liverun_indoor/left",
      "rear": "Dataset/liverun_indoor/rear",
      "right": "Dataset/liverun_indoor/right",
    }
    processor = ImageProcessor(video_paths, img_car)
    processor.run()
