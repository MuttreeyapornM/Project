# --------------------------------Test_cam--------------------------------
# import cv2

# # กำหนด ID ของกล้องที่ต้องการเปิด
# camera_ids = [0, 2, 4, 6] 
# captures = []

# # เปิดกล้องแต่ละตัว
# for cam_id in camera_ids:
#     cap = cv2.VideoCapture(cam_id)
#     if cap.isOpened():
#         captures.append((cam_id, cap))
#     else:
#         print(f"ไม่สามารถเปิดกล้องที่ ID {cam_id}")

# # ตรวจสอบว่ามีกล้องที่เปิดสำเร็จหรือไม่
# if not captures:
#     print("ไม่สามารถเปิดกล้องได้เลย")
#     exit()

# while True:
#     frames = []
#     for cam_id, cap in captures:
#         ret, frame = cap.read()
#         if ret:
#             frames.append((cam_id, frame))
#         else:
#             print(f"ไม่สามารถอ่านข้อมูลจากกล้อง ID {cam_id}")

#     # แสดงผลลัพธ์จากแต่ละกล้อง
#     for cam_id, frame in frames:
#         cv2.imshow(f'Camera {cam_id}', frame)

#     # กด 'q' เพื่อออก
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ปิดกล้องทั้งหมด
# for _, cap in captures:
#     cap.release()
# cv2.destroyAllWindows()



# #--------------------------------------- save the picture ----------------------------------------
import cv2
import numpy as np
import os
import time

# สร้างโฟลเดอร์สำหรับแต่ละกล้อง
output_folders = ["camera1_images", "camera2_images", "camera3_images", "camera4_images"]
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# เปิดกล้อง 4 ตัว (กล้อง 0, 2, 4, 6)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(4)
cap4 = cv2.VideoCapture(6)

# ตรวจสอบว่ากล้องถูกเปิดหรือไม่
if not all(cap.isOpened() for cap in [cap1, cap2, cap3, cap4]):
    print("ไม่สามารถเชื่อมต่อกับกล้องได้")
    exit()

# กำหนด resolution ของกล้องเป็น 1280x720
for cap in [cap1, cap2, cap3, cap4]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # อ่านภาพจากกล้องทั้ง 4 ตัว
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # ถ้าอ่านภาพได้สำเร็จจากทุกกล้อง
    if ret1 and ret2 and ret3 and ret4:
        # ลดขนาดเฉพาะภาพที่ "แสดง" เป็น 640x360 เพื่อรวมกัน
        display_frame1 = cv2.resize(frame1, (640, 360))
        display_frame2 = cv2.resize(frame2, (640, 360))
        display_frame3 = cv2.resize(frame3, (640, 360))
        display_frame4 = cv2.resize(frame4, (640, 360))

        # รวมภาพจากทั้ง 4 กล้องในรูปแบบ 2x2
        top_row = np.hstack((display_frame1, display_frame2))  # แถวบน
        bottom_row = np.hstack((display_frame3, display_frame4))  # แถวล่าง
        final_display = np.vstack((top_row, bottom_row))  # รวมแถวเข้าด้วยกัน

        # แสดงภาพที่รวมจากทั้ง 4 กล้อง
        cv2.imshow('4 Camera Views (Resized Display)', final_display)

        # บันทึกภาพจากแต่ละกล้องที่ความละเอียด 1280x720
        timestamp = int(time.time())  # ใช้เวลาเป็นชื่อไฟล์
        cv2.imwrite(os.path.join(output_folders[0], f"frame_{timestamp}.jpg"), frame1)
        cv2.imwrite(os.path.join(output_folders[1], f"frame_{timestamp}.jpg"), frame2)
        cv2.imwrite(os.path.join(output_folders[2], f"frame_{timestamp}.jpg"), frame3)
        cv2.imwrite(os.path.join(output_folders[3], f"frame_{timestamp}.jpg"), frame4)

    # ออกจากลูปเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกล้องและหน้าต่างทั้งหมด
for cap in [cap1, cap2, cap3, cap4]:
    cap.release()
cv2.destroyAllWindows()


#-----------------------------save the video-----------------------------------
# import cv2
# import numpy as np
# import os
# import time

# # สร้างโฟลเดอร์สำหรับบันทึกวิดีโอของแต่ละกล้อง
# output_folders = ["camera1_videos", "camera2_videos", "camera3_videos", "camera4_videos"]
# for folder in output_folders:
#     os.makedirs(folder, exist_ok=True)

# # เปิดกล้อง 4 ตัว (กล้อง 0, 2, 4, 6)
# cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(2)
# cap3 = cv2.VideoCapture(4)
# cap4 = cv2.VideoCapture(6)

# # ตรวจสอบว่ากล้องถูกเปิดหรือไม่
# if not all(cap.isOpened() for cap in [cap1, cap2, cap3, cap4]):
#     print("ไม่สามารถเชื่อมต่อกับกล้องได้")
#     exit()

# # กำหนด resolution ของกล้องเป็น 1280x720
# for cap in [cap1, cap2, cap3, cap4]:
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # สร้าง VideoWriter สำหรับบันทึกไฟล์ .mp4
# fps = 10  # อัตราเฟรม (frames per second)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ใช้ codec สำหรับไฟล์ .mp4

# # ตั้งชื่อไฟล์วิดีโอโดยใช้ timestamp
# timestamp = int(time.time())
# video_writers = [
#     cv2.VideoWriter(os.path.join(output_folders[0], f"camera1_{timestamp}.mp4"), fourcc, fps, (1280, 720)),
#     cv2.VideoWriter(os.path.join(output_folders[1], f"camera2_{timestamp}.mp4"), fourcc, fps, (1280, 720)),
#     cv2.VideoWriter(os.path.join(output_folders[2], f"camera3_{timestamp}.mp4"), fourcc, fps, (1280, 720)),
#     cv2.VideoWriter(os.path.join(output_folders[3], f"camera4_{timestamp}.mp4"), fourcc, fps, (1280, 720))
# ]

# while True:
#     # อ่านภาพจากกล้องทั้ง 4 ตัว
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()
#     ret3, frame3 = cap3.read()
#     ret4, frame4 = cap4.read()

#     # ถ้าอ่านภาพได้สำเร็จจากทุกกล้อง
#     if ret1 and ret2 and ret3 and ret4:
#         # ลดขนาดเฉพาะภาพที่ "แสดง" เป็น 640x360 เพื่อรวมกัน
#         display_frame1 = cv2.resize(frame1, (640, 360))
#         display_frame2 = cv2.resize(frame2, (640, 360))
#         display_frame3 = cv2.resize(frame3, (640, 360))
#         display_frame4 = cv2.resize(frame4, (640, 360))

#         # รวมภาพจากทั้ง 4 กล้องในรูปแบบ 2x2
#         top_row = np.hstack((display_frame1, display_frame2))  # แถวบน
#         bottom_row = np.hstack((display_frame3, display_frame4))  # แถวล่าง
#         final_display = np.vstack((top_row, bottom_row))  # รวมแถวเข้าด้วยกัน

#         # แสดงภาพที่รวมจากทั้ง 4 กล้อง
#         cv2.imshow('4 Camera Views (Resized Display)', final_display)

#         # บันทึกวิดีโอของแต่ละกล้องที่ความละเอียด 1280x720
#         video_writers[0].write(frame1)
#         video_writers[1].write(frame2)
#         video_writers[2].write(frame3)
#         video_writers[3].write(frame4)

#     # ออกจากลูปเมื่อกด 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ปิดการเชื่อมต่อกล้องและหน้าต่างทั้งหมด
# for cap in [cap1, cap2, cap3, cap4]:
#     cap.release()

# # ปิดไฟล์วิดีโอ
# for writer in video_writers:
#     writer.release()

# cv2.destroyAllWindows()


#------------------------------id cam test-------------------------------------
# import cv2

# def find_camera_ids():
#     available_cameras = []
#     # ลอง ID กล้องตั้งแต่ 0 ถึง 9 (คุณสามารถปรับช่วงนี้ได้ตามจำนวนกล้องที่คุณคาดหวัง)
#     for i in range(10):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 print(f"✅ Camera ID {i} is working. Frame shape: {frame.shape}")
#                 available_cameras.append(i)
#                 cap.release()
#             else:
#                 print(f"❌ Camera ID {i} found, but cannot read frame.")
#                 cap.release()
#         else:
#             print(f"--- Camera ID {i} is not available.")
#     return available_cameras

# if __name__ == "__main__":
#     print("Searching for available camera IDs...")
#     ids = find_camera_ids()
#     if ids:
#         print(f"\nAvailable camera IDs: {ids}")
#     else:
#         print("\nNo cameras found.")