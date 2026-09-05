import os
import threading

import cv2
import numpy as np

try:
    import kornia
    import torch

    KORNIA_AVAILABLE = True
except ImportError:
    kornia = None
    torch = None
    KORNIA_AVAILABLE = False

try:
    from image_processing import ImageAdjuster, ImageStitcher
    from param_settings import Car_dst_points, img_car, total_h, total_w

    BEV_AVAILABLE = True
except ImportError:
    ImageAdjuster = None
    ImageStitcher = None
    Car_dst_points = None
    img_car = None
    total_w, total_h = 1280, 720
    BEV_AVAILABLE = False


class BEVProcessor:
    def __init__(
        self,
        video_paths,
        img_car=None,
        display_width=800,
        display_height=600,
        map_width=None,
        map_height=None,
    ):
        self.video_paths = video_paths
        self.car = img_car
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width if map_width else total_w
        self.map_height = map_height if map_height else total_h
        self.caps = {k: self._open_cap(v) for k, v in video_paths.items()}
        self.calibration_data = {cam: self._load_calib(cam) for cam in video_paths}
        if torch is not None:
            self.torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.torch_device = None
        self.cuda_available = (
            torch is not None and torch.cuda.is_available() and KORNIA_AVAILABLE
        )
        self._H_tensors = {}
        self._gpu_lock = threading.Lock()

    def _open_cap(self, path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _load_calib(self, cameraID):
        yaml_filename = os.path.join("yaml", f"calibration_data_{cameraID}.yaml")
        if not os.path.exists(yaml_filename):
            return {
                "camera_matrix": np.eye(3, dtype=np.float32),
                "dist_coeffs": np.zeros((1, 5), dtype=np.float32),
                "homography": np.eye(3, dtype=np.float32),
            }
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        data = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
            "homography": fs.getNode("homography").mat(),
        }
        fs.release()
        return data

    def _get_H_tensor(self, cameraID):
        if cameraID not in self._H_tensors:
            H = self.calibration_data[cameraID]["homography"]
            self._H_tensors[cameraID] = (
                torch.from_numpy(H).unsqueeze(0).float().to(self.torch_device)
            )
        return self._H_tensors[cameraID]

    def _warp_gpu(self, image, cameraID):
        with self._gpu_lock:
            img_t = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.torch_device)
            )
            with torch.no_grad():
                out = kornia.geometry.transform.warp_perspective(
                    img_t,
                    self._get_H_tensor(cameraID),
                    dsize=(self.map_height, self.map_width),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            return (
                out.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )

    def process_image(self, image, cameraID):
        calib = self.calibration_data[cameraID]
        undis = cv2.undistort(image, calib["camera_matrix"], calib["dist_coeffs"])
        if cameraID in ["rear", "right"]:
            undis = cv2.rotate(undis, cv2.ROTATE_180)
        warped = (
            self._warp_gpu(undis, cameraID)
            if self.cuda_available
            else cv2.warpPerspective(
                undis, calib["homography"], (self.map_width, self.map_height)
            )
        )
        return undis, warped

    def grab_frame(self, cam_id, cap, images, warped_list, idx):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return False
        undis, warped = self.process_image(frame, cam_id)
        images[idx] = undis
        warped_list[idx] = warped
        return True

    def get_bev_frame(self, include_display=True):
        cam_names = ["Front", "Left", "Rear", "Right"]
        images = [None] * len(self.caps)
        warped = [None] * len(self.caps)
        threads = [
            threading.Thread(
                target=self.grab_frame, args=(cam_id, cap, images, warped, i)
            )
            for i, (cam_id, cap) in enumerate(self.caps.items())
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if all(img is not None for img in images):
            if (
                BEV_AVAILABLE
                and ImageStitcher is not None
                and hasattr(ImageStitcher, "get_weights_and_masks_liverun")
            ):
                merged = ImageStitcher.get_weights_and_masks_liverun(warped)
                if (
                    self.car is not None
                    and ImageAdjuster is not None
                    and hasattr(ImageAdjuster, "overlay_image_perspective")
                ):
                    merged = ImageAdjuster.overlay_image_perspective(
                        merged.copy(), self.car, Car_dst_points
                    )
            else:
                merged = self.simple_stitch(warped)

            if include_display:
                rw, rh = self.display_width // 2, self.display_height // 2
                resized = [cv2.resize(img, (rw, rh)) for img in images]
                for i, img in enumerate(resized):
                    cv2.putText(
                        img,
                        cam_names[i],
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                top = np.hstack((resized[0], resized[1]))
                bottom = np.hstack((resized[2], resized[3]))
                return merged, np.vstack((top, bottom))
            return merged, None
        return None, None

    def simple_stitch(self, warped_images):
        result = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        for warped in warped_images:
            if warped is not None:
                mask = (warped > 0).astype(np.float32)
                result = (result * (1 - mask) + warped * mask).astype(np.uint8)
        return result

    def release(self):
        for cap in self.caps.values():
            cap.release()
