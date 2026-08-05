# ============================================================
#  computer1_bev_server.py  —  BEV Image Capture & WebSocket Server
#  คอมพิวเตอร์ที่ 1: รับภาพจากกล้อง 4 ตัว → รวมเป็น BEV → ส่งผ่าน WebSocket
# ============================================================

import os
import cv2
import time
import asyncio
import threading
import numpy as np
import argparse
import base64
import json
import websockets
from websockets.server import serve

# ── BEV Modules ─────────────────────────────────────────────
try:
    import kornia
    import torch
    KORNIA_AVAILABLE = True
    print("✅ Kornia available - using GPU warpPerspective")
except ImportError:
    KORNIA_AVAILABLE = False
    torch = None
    print("⚠️  Kornia not found. Falling back to CPU warpPerspective")

try:
    from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
    from param_settings import img_car, Car_dst_points, total_w, total_h
    BEV_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: BEV modules not found. Using simple stitch fallback.")
    BEV_AVAILABLE = False
    img_car = None
    Car_dst_points = None
    total_w, total_h = 1280, 720


# ═══════════════════════════════════════════════════════════
#  BEVProcessor (เหมือนเดิม)
# ═══════════════════════════════════════════════════════════

class BEVProcessor:
    def __init__(self, video_paths, img_car=None, display_width=800, display_height=600,
                 map_width=None, map_height=None):
        self.video_paths    = video_paths
        self.car            = img_car
        self.display_width  = display_width
        self.display_height = display_height
        self.map_width      = map_width  if map_width  else total_w
        self.map_height     = map_height if map_height else total_h
        self.caps           = {k: self._open_cap(v) for k, v in video_paths.items()}
        self.calibration_data = {cam: self._load_calib(cam) for cam in video_paths}
        if torch is not None:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = None
        self.cuda_available = (torch is not None and torch.cuda.is_available() and KORNIA_AVAILABLE)
        self._H_tensors = {}
        self._gpu_lock = threading.Lock()   # Kornia/torch.linalg ไม่ thread-safe

    def _open_cap(self, path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _load_calib(self, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        if not os.path.exists(yaml_filename):
            return {"camera_matrix": np.eye(3, dtype=np.float32),
                    "dist_coeffs":   np.zeros((1, 5), dtype=np.float32),
                    "homography":    np.eye(3, dtype=np.float32)}
        fs   = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        data = {"camera_matrix": fs.getNode("camera_matrix").mat(),
                "dist_coeffs":   fs.getNode("dist_coeffs").mat(),
                "homography":    fs.getNode("homography").mat()}
        fs.release()
        return data

    def _get_H_tensor(self, cameraID):
        if cameraID not in self._H_tensors:
            H = self.calibration_data[cameraID]["homography"]
            self._H_tensors[cameraID] = (
                torch.from_numpy(H).unsqueeze(0).float().to(self.torch_device))
        return self._H_tensors[cameraID]

    def _warp_gpu(self, image, cameraID):
        # Kornia ใช้ torch.linalg.inv ภายใน ซึ่งไม่ thread-safe → ต้องใช้ lock
        with self._gpu_lock:
            img_t = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                     .float().to(self.torch_device))
            with torch.no_grad():
                out = kornia.geometry.transform.warp_perspective(
                    img_t, self._get_H_tensor(cameraID),
                    dsize=(self.map_height, self.map_width),
                    mode='bilinear', padding_mode='zeros', align_corners=False)
            return out.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    def process_image(self, image, cameraID):
        calib = self.calibration_data[cameraID]
        undis = cv2.undistort(image, calib["camera_matrix"], calib["dist_coeffs"])
        if cameraID in ["rear", "right"]:
            undis = cv2.rotate(undis, cv2.ROTATE_180)
        warped = (self._warp_gpu(undis, cameraID) if self.cuda_available
                  else cv2.warpPerspective(undis, calib["homography"],
                                           (self.map_width, self.map_height)))
        return undis, warped

    def grab_frame(self, cam_id, cap, images, warped_list, idx):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return False
        undis, warped = self.process_image(frame, cam_id)
        images[idx]      = undis
        warped_list[idx] = warped
        return True

    def get_bev_frame(self, include_display=True):
        cam_names = ["Front", "Left", "Rear", "Right"]
        images  = [None] * len(self.caps)
        warped  = [None] * len(self.caps)
        threads = [threading.Thread(target=self.grab_frame,
                   args=(cam_id, cap, images, warped, i))
                   for i, (cam_id, cap) in enumerate(self.caps.items())]
        for t in threads: t.start()
        for t in threads: t.join()

        if all(img is not None for img in images):
            if BEV_AVAILABLE and hasattr(ImageStitcher, 'get_weights_and_masks_liverun'):
                merged = ImageStitcher.get_weights_and_masks_liverun(warped)
                if self.car is not None and hasattr(ImageAdjuster, 'overlay_image_perspective'):
                    merged = ImageAdjuster.overlay_image_perspective(
                        merged.copy(), self.car, Car_dst_points)
            else:
                merged = self.simple_stitch(warped)

            if include_display:
                rw, rh = self.display_width // 2, self.display_height // 2
                resized = [cv2.resize(img, (rw, rh)) for img in images]
                for i, img in enumerate(resized):
                    cv2.putText(img, cam_names[i], (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                top    = np.hstack((resized[0], resized[1]))
                bottom = np.hstack((resized[2], resized[3]))
                return merged, np.vstack((top, bottom))
            return merged, None
        return None, None

    def simple_stitch(self, warped_images):
        result = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        for w in warped_images:
            if w is not None:
                mask   = (w > 0).astype(np.float32)
                result = (result * (1 - mask) + w * mask).astype(np.uint8)
        return result

    def release(self):
        for cap in self.caps.values():
            cap.release()


# ═══════════════════════════════════════════════════════════
#  WebSocket Server
# ═══════════════════════════════════════════════════════════

class BEVStreamServer:
    def __init__(self, bev_processor, host="0.0.0.0", port=8765,
                 jpeg_quality=85, target_fps=15, show_preview=False):
        self.bev_processor = bev_processor
        self.host          = host
        self.port          = port
        self.jpeg_quality  = jpeg_quality
        self.target_fps    = target_fps
        self.show_preview  = show_preview
        self.clients       = set()
        self.latest_frame  = None
        self.frame_lock    = asyncio.Lock()
        self.running       = True
        self.frame_count   = 0

    def encode_frame(self, frame):
        """เข้ารหัส frame เป็น JPEG base64 สำหรับส่งผ่าน WebSocket"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    async def capture_loop(self):
        """Loop จับภาพและอัปเดต latest_frame"""
        interval = 1.0 / self.target_fps
        print(f"📷 Capture loop started @ {self.target_fps} FPS")
        while self.running:
            t_start = time.time()
            bev_frame, cam_display = self.bev_processor.get_bev_frame()

            if bev_frame is not None:
                async with self.frame_lock:
                    self.latest_frame = bev_frame.copy()
                self.frame_count += 1

                if self.show_preview:
                    h_b, w_b = bev_frame.shape[:2]
                    cv2.imshow("BEV [Computer 1]",
                               cv2.resize(bev_frame, (w_b // 2, h_b // 2)))
                    if cam_display is not None:
                        h_c, w_c = cam_display.shape[:2]
                        cv2.imshow("Cameras [Computer 1]",
                                   cv2.resize(cam_display, (w_c // 2, h_c // 2)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                if self.frame_count % 30 == 0:
                    print(f"  📦 Captured {self.frame_count} frames | "
                          f"Clients: {len(self.clients)}")

            elapsed = time.time() - t_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def broadcast_loop(self):
        """Loop ส่ง frame ไปยัง clients ทุกคน"""
        interval = 1.0 / self.target_fps
        while self.running:
            t_start = time.time()

            if self.clients:
                async with self.frame_lock:
                    frame = self.latest_frame

                if frame is not None:
                    try:
                        encoded = self.encode_frame(frame)
                        h, w    = frame.shape[:2]
                        payload = json.dumps({
                            "type":      "bev_frame",
                            "width":     w,
                            "height":    h,
                            "timestamp": time.time(),
                            "frame":     encoded
                        })
                        # ส่งพร้อมกันทุก client
                        if self.clients:
                            await asyncio.gather(
                                *[client.send(payload) for client in list(self.clients)],
                                return_exceptions=True
                            )
                    except Exception as e:
                        print(f"⚠️  Broadcast error: {e}")

            elapsed = time.time() - t_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    # async def handle_client(self, websocket):
    #     """จัดการ client แต่ละราย"""
    #     client_addr = websocket.remote_address
    #     print(f"🔌 Client connected: {client_addr}")
    #     self.clients.add(websocket)
    #     try:
    #         # ส่ง metadata ให้ client
    #         async with self.frame_lock:
    #             frame = self.latest_frame
    #         if frame is not None:
    #             h, w = frame.shape[:2]
    #             await websocket.send(json.dumps({
    #                 "type":       "metadata",
    #                 "bev_width":  w,
    #                 "bev_height": h,
    #                 "fps":        self.target_fps
    #             }))

    #         # รอรับ message (ping/control) จาก client
    #         async for message in websocket:
    #             data = json.loads(message)
    #             if data.get("type") == "ping":
    #                 await websocket.send(json.dumps({"type": "pong",
    #                                                  "timestamp": time.time()}))
    #     except websockets.exceptions.ConnectionClosed:
    #         print(f"🔌 Client disconnected: {client_addr}")
    #     except Exception as e:
    #         print(f"⚠️  Client error {client_addr}: {e}")
    #     finally:
    #         self.clients.discard(websocket)

    # -------------------------------------------------ใหม่-----------------------------------
    # ------------------------------------------------- FIXED -----------------------------------

    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        print(f"🔌 Client connected: {client_addr}")
        self.clients.add(websocket)

        try:
            # ส่ง metadata ทันที (กัน timeout)
            await websocket.send(json.dumps({
                "type": "metadata",
                "bev_width": 0,
                "bev_height": 0,
                "fps": self.target_fps
            }))

            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))

        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_addr}")
        except Exception as e:
            print(f"⚠️  Client error {client_addr}: {e}")
        finally:
            self.clients.discard(websocket)
   
    async def run(self):
        print(f"🚀 BEV WebSocket Server starting on ws://{self.host}:{self.port}")
        async with serve(self.handle_client, self.host, self.port):
            await asyncio.gather(
                self.capture_loop(),
                self.broadcast_loop()
            )
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════
#  Argparser
# ═══════════════════════════════════════════════════════════

def get_argparser():
    parser = argparse.ArgumentParser(description="Computer 1: BEV Capture & WebSocket Server")
    parser.add_argument("--front_cam",      type=int,   default=2,        help="Front camera index")
    parser.add_argument("--left_cam",       type=int,   default=4,        help="Left camera index")
    parser.add_argument("--rear_cam",       type=int,   default=0,        help="Rear camera index")
    parser.add_argument("--right_cam",      type=int,   default=6,        help="Right camera index")
    parser.add_argument("--host",           type=str,   default="0.0.0.0",help="WebSocket server host")
    parser.add_argument("--port",           type=int,   default=8765,     help="WebSocket server port")
    parser.add_argument("--fps",            type=int,   default=15,       help="Target FPS to stream")
    parser.add_argument("--jpeg_quality",   type=int,   default=85,       help="JPEG quality (1-100)")
    parser.add_argument("--display_width",  type=int,   default=800)
    parser.add_argument("--display_height", type=int,   default=600)
    parser.add_argument("--show_preview",   action='store_true', default=False)
    return parser


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    opts = get_argparser().parse_args()

    video_paths = {
        "front": opts.front_cam,
        "left":  opts.left_cam,
        "rear":  opts.rear_cam,
        "right": opts.right_cam,
    }

    print("=" * 60)
    print("🖥️  COMPUTER 1 — BEV Capture Server")
    print(f"   Cameras: {video_paths}")
    print(f"   Stream : ws://{opts.host}:{opts.port}")
    print(f"   FPS    : {opts.fps} | JPEG quality: {opts.jpeg_quality}")
    print("=" * 60)

    bev_processor = BEVProcessor(
        video_paths,
        img_car        = img_car if BEV_AVAILABLE else None,
        display_width  = opts.display_width,
        display_height = opts.display_height,
    )

    server = BEVStreamServer(
        bev_processor,
        host          = opts.host,
        port          = opts.port,
        jpeg_quality  = opts.jpeg_quality,
        target_fps    = opts.fps,
        show_preview  = opts.show_preview,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        server.running = False
        bev_processor.release()
        print("✅ Server stopped")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────
# วิธีรัน:
#   pip install websockets opencv-python numpy kornia torch
#   source venv/bin/activate
#   python sep_com1.py --front_cam 2 --left_cam 6 --rear_cam 0 --right_cam 4 --host 192.168.1.100 --port 8765 --fps 15 --jpeg_quality 85 --show_preview
#   python sep_com1.py --front_cam 0 --left_cam 4 --rear_cam 2 --right_cam 6 --host 0.0.0.0 --port 8765 --fps 15 --jpeg_quality 85 --show_preview

# ─────────────────────────────────────────────────────────────
