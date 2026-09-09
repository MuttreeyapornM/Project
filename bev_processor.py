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
    from capture_settings import configure_capture, is_camera_source
except ImportError:  # keep the module importable if the helper is absent
    configure_capture = None
    is_camera_source = None

try:
    import camera_grabber
    from camera_grabber import CameraGrabber
except ImportError:
    camera_grabber = None
    CameraGrabber = None

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

CAP_W, CAP_H = 1280, 720


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
        # Live cameras get a grabber thread so the driver queue cannot back up.
        # Video files are left alone: a grabber would race through the file.
        self.grabbers = {}
        if CameraGrabber is not None and not camera_grabber.DISABLED:
            for k, v in video_paths.items():
                if is_camera_source is not None and is_camera_source(v):
                    self.grabbers[k] = CameraGrabber(self.caps[k], name=k).start()
        self.calibration_data = {cam: self._load_calib(cam) for cam in video_paths}
        # The cameras are rigidly mounted, so undistortion and the BEV homography
        # are fixed geometry. Build the remap lookup tables ONCE here instead of
        # letting cv2.undistort rebuild them on every frame (~40 ms/camera).
        self.maps = {cam: self._build_maps(cam) for cam in video_paths}

    def _open_cap(self, path):
        cap = cv2.VideoCapture(path)
        if configure_capture is not None:
            # Requests MJPG on live cameras. Without it V4L2 negotiates YUYV,
            # which caps these USB 2.0 cameras at 10 fps at 720p regardless of
            # how fast the rest of the pipeline runs. See capture_settings.py.
            return configure_capture(cap, path, CAP_W, CAP_H)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
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

    def _build_maps(self, cameraID):
        """Precompute two LUTs per camera.

        "und": undistorted view (rotated 180 for rear/right, matching the old
               cv2.undistort + cv2.rotate output).
        "bev": undistort + rotate + homography composed into ONE map, so the
               BEV image is produced by a single cv2.remap with a single
               resampling instead of undistort followed by warpPerspective.
        """
        calib = self.calibration_data[cameraID]
        K = np.asarray(calib["camera_matrix"], dtype=np.float64)
        dist = np.asarray(calib["dist_coeffs"], dtype=np.float64)
        H = np.asarray(calib["homography"], dtype=np.float64)

        # Float maps: for undistorted pixel (u,v), the source pixel is
        # (m1[v,u], m2[v,u]). Same K as new camera matrix = cv2.undistort default.
        m1, m2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (CAP_W, CAP_H), cv2.CV_32FC1
        )

        rotated = cameraID in ("rear", "right")

        # --- map for the undistorted output ---
        if rotated:
            # rotate180(remap(img, m)) == remap(img, m flipped in both axes)
            u1 = np.ascontiguousarray(m1[::-1, ::-1])
            u2 = np.ascontiguousarray(m2[::-1, ::-1])
        else:
            u1, u2 = m1, m2
        und_map = cv2.convertMaps(u1, u2, cv2.CV_16SC2)

        # --- composed BEV map ---
        # BEV output grid -> (rotated) undistorted image coords via H^-1
        Hinv = np.linalg.inv(H)
        us, vs = np.meshgrid(
            np.arange(self.map_width, dtype=np.float64),
            np.arange(self.map_height, dtype=np.float64),
        )
        denom = Hinv[2, 0] * us + Hinv[2, 1] * vs + Hinv[2, 2]
        x = (Hinv[0, 0] * us + Hinv[0, 1] * vs + Hinv[0, 2]) / denom
        y = (Hinv[1, 0] * us + Hinv[1, 1] * vs + Hinv[1, 2]) / denom
        if rotated:
            # H was calibrated against the rotated undistorted image; undo the
            # rotation before looking up the undistort map
            x = (CAP_W - 1) - x
            y = (CAP_H - 1) - y
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # compose: sample the undistort maps at the homography's source coords;
        # points outside the camera image get -1 -> remap paints them black
        bx = cv2.remap(
            m1, x, y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
        )
        by = cv2.remap(
            m2, x, y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
        )
        bev_map = cv2.convertMaps(bx, by, cv2.CV_16SC2)

        return {"und": und_map, "bev": bev_map}

    def process_image(self, image, cameraID, want_undistorted=True):
        """Return (undistorted, bev).

        The undistorted view is only used for the debug display grid, so when
        that is off we skip its remap entirely -- roughly a third of the
        per-camera warping cost for an image the navigation code never reads.
        """
        bev_map = self.maps[cameraID]["bev"]
        warped = cv2.remap(image, bev_map[0], bev_map[1], cv2.INTER_LINEAR)
        undis = None
        if want_undistorted:
            und_map = self.maps[cameraID]["und"]
            undis = cv2.remap(image, und_map[0], und_map[1], cv2.INTER_LINEAR)
        return undis, warped

    def grab_frame(self, cam_id, cap, images, warped_list, idx, want_undistorted=True):
        grabber = self.grabbers.get(cam_id)
        if grabber is not None:
            # Newest frame, never a stale queue entry.
            ret, frame = grabber.read()
            if not ret:
                return False
        else:
            ret, frame = cap.read()
            if not ret:
                # Video-file source: loop back to the start.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    return False
        undis, warped = self.process_image(frame, cam_id, want_undistorted)
        images[idx] = undis if undis is not None else True  # presence marker
        warped_list[idx] = warped
        return True

    def get_bev_frame(self, include_display=True):
        cam_names = ["Front", "Left", "Rear", "Right"]
        images = [None] * len(self.caps)
        warped = [None] * len(self.caps)
        threads = [
            threading.Thread(
                target=self.grab_frame,
                args=(cam_id, cap, images, warped, i, include_display),
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

    def release(self):
        """Stop grabber threads and release the captures."""
        for g in self.grabbers.values():
            g.release()
        self.grabbers = {}
        for cap in self.caps.values():
            try:
                cap.release()
            except Exception:
                pass

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
