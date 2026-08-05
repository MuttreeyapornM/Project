#---------------------env activate-------------------
windows : myenv\Scripts\activate
Linux : source myenv/bin/activate
for main folder: source venv/bin/activate

source /opt/ros/foxy/setup.bash




#---------------------Testing------------------------
1. single image : 
    python image_test.py --input image.png  --dataset custom --model deeplabv3plus_mobilenet --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16.pth --save_val_results_to test_results

2. folder image : 
    python image_test.py --input datasets/3  --dataset custom --model deeplabv3plus_mobilenet --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16.pth --save_val_results_to test_results

3. Real-Time : 
    python RealTime.py ---> only real-time input without model processing

4. video : 
    python video_test.py --input video5fps.mp4 --output test_results --ckpt checkpoints/latest_deeplabv3plus_mobilenet_custom_os16.pth --dataset custom

5. ...line.py : 
    python leftline.py --input video5fps.mp4 --dataset custom --ckpt ./checkpoints/clean_state_dict.pth --output test_results --nav_line_color green --nav_line_thickness 3 --fps 5

6. navigate.py ---> Real-Time BEV : python navigate.py --use_camera --front_cam 2 --left_cam 4 --rear_cam 0 --right_cam 6 --ckpt ./checkpoints/clean_state_dict.pth

6. navigate.py ---> Real-Time BEV : python navigate.py --use_camera --front_cam 2 --left_cam 4 --right_cam 6 --ckpt ./checkpoints/clean_state_dict.pth



6. navigate.py ---> video input : python navigate.py --input video5fps.mp4 --output test_results --ckpt ./checkpoints/clean_state_dict.pth


#    After clean weight
    python leftline.py --input video5fps.mp4 --dataset custom --ckpt ./checkpoints/clean_state_dict.pth --output test_results --nav_line_color green --nav_line_thickness 3 --fps 5

    

# NOTE
1. # navigate1.py
navigate1.py --> original navigation without ros + one page only output
📌 Line 1: จุด offset จากขอบซ้ายเข้ามาในเลน
Point 1: x=566, y=0
Point 2: x=547, y=197
Point 3: x=542, y=394
Point 4: x=344, y=591
Point 5: x=418, y=788
Point 6: x=361, y=985
Point 7: x=620, y=1182
Point 8: x=338, y=1379

📌 Line 2: จุดกึ่งกลางของเลน
Point 1: x=616, y=552
Point 2: x=624, y=828

2. # navigate0.py 
navigate0.py --> only model processing and one page output 

3. # navigate_copy.py + ros_point_publisher.py + pure_pursuit_controller_copy.py
navigate_ros --> combind with ros2 ---> ros node sent value from 1-2-3-4
📌 Line 1: Destination Point
Point 1: x=250, y=0
Point 2: x=672, y=153
Point 3: x=576, y=306
Point 4: x=645, y=459

📌 Line 2: Current Point
Point 1: x=630, y=695

4. # navigate.py + ros_point_publisher.py + pure_pursuit_controller.py/pure_pursuit_controller_filtered.py
navigate --> 1 page output + video input ---> ros node sent value from 4-3-2-1
📌 Line 1: Destination Point
Point 4: x=667, y=459
Point 3: x=358, y=306
Point 2: x=581, y=153
Point 1: x=250, y=0

📌 Line 2: Current Point
Point 1: x=607, y=657




5.  # detecte.py + ros2_publish.py ---> only video input : publish id of detection
    detect.py --> focus on id object detection of ROI 1/3 from top frame 
    video : python detect.py --input video5fps.mp4 --show_preview --show_mask_ids
    +
    ros2 topic echo /roi_mask_ids


6.  # object_ros2_enabled.py---> 1/3 of top frame processing and id detecteion to cmd_vel(speed)
python object_ros2_enabled.py --use_camera --front_cam 0 --left_cam 2 --rear_cam 6 --right_cam 4 --use_roi --show_mask_ids --ckpt ./checkpoints/best2700.pth 
ros2 topic echo /cmd_vel
python object_ros2_enabled.py --input video5fps.mp4 --show_mask_ids --use_roi --ckpt ./checkpoints/best2700.pth 


7. # object.py ----> มีหน้า output แต่ไม่มีค่า cmd_vel ออกมา ไม่มีท็อปปิคด้วย
id_publisher.py ----> ไม่มีหน้า output






#---------------------------master's degree----------------------------------------

8. # create_point.py --> copy of navigate.py for create ROI 1/2 frame and new solution for create point
cameras input : python create_point.py --use_camera --front_cam 0 --left_cam 2 --rear_cam 6 --right_cam 4 --ckpt ./checkpoints/best2700.pth 
video input : python create_point.py --input video5fps.mp4 --output test_results --ckpt ./checkpoints/best2700.pth --output test_results
#-------------------ROS2--------------------------------------------

1. run real-time
    open terminal
    source venv/bin/activate
    source /opt/ros/foxy/setup.bash
    python navigate.py --use_camera --front_cam 0 --left_cam 2 --rear_cam 6 --right_cam 4 --ckpt ./checkpoints/best2700.pth 
    python _____.py --input video5fps.mp4 --output test_results --ckpt ./checkpoints/best2700.pth 

2. pure_pursuit_controller.py
    source /opt/ros/foxy/setup.bash
    python pure_pursuit_controller.py
    python pure_pursuit_controller_copy.py
    python pure_pursuit_controller_filtered.py

    # ดู topics ที่มี
    ros2 topic list

    # ดูข้อมูลที่รับ (คุณใช้อยู่แล้ว)
    ros2 topic echo /destination_path
    ros2 topic echo /current_path

    # ดูคำสั่งที่ส่งไปยัง drive-by-wire
    ros2 topic echo /vehicle/steering_cmd
    ros2 topic echo /vehicle/throttle_cmd
    ros2 topic echo /vehicle/brake_cmd
    ros2 topic echo /cmd_vel








## MASTER's new version

## NEW VERSION
# REAL-TIME TEST#1 WITH ROS2

# -----------------------------------------------SERVER---------------------------------------
# ─────────────────────────────────────────────────────────────
# วิธีรัน:
#   pip install websockets opencv-python numpy kornia torch
    source venv/bin/activate
    python sep_com1.py --front_cam 2 --left_cam 6 --rear_cam 0 --right_cam 4 --host 192.168.1.100 --port 8765 --fps 15 --jpeg_quality 85 --show_preview
    python sep_com1.py --front_cam 0 --left_cam 4 --rear_cam 2 --right_cam 6 --host 0.0.0.0 --port 8765 --fps 15 --jpeg_quality 85 --show_preview

# ─────────────────────────────────────────────────────────────
#   Video Simulation
    python sep_com2_video_sim.py \
    --video_input lx.mp4 \
    --output ./output\
    --dataset custom \
    --ckpt ./checkpoints_new/iter_90000_deeplabv3plus_mobilenet_custom_os16.pth \
    --enable_ros2

# เพิ่ม CBF เข้าไป
    python sep_sim.py \
    --video_input video5fps1.mp4 \
    --output ./output\
    --dataset custom \
    --ckpt ./checkpoints/best3700.pth 

# ใส่ CBF พวงมาลัยให้สมูทขึ้น
    python sim_copy.py \
    --video_input video5fps1.mp4 \
    --output ./output\
    --dataset custom \
    --ckpt ./checkpoints/best3700.pth \
    --enable_ros2 \
    --steering_filter_alpha 0.8 \
    --steering_deadband_deg 1.0 \
    --steering_max_rate_deg_s 10.0

# ---------------------------------REAL-TIME-----------------------------------------------------
# ============================================================
#  sep_com_combined.py
#  Single-computer pipeline:
#  cameras -> BEV -> segmentation/navigation -> /cmd_vel
# ============================================================

python sep_com_combined_timing.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints/best3700.pth \
    --enable_ros2 \
    --cmd_vel_publish_hz 30 \
    --fp16 \
    --async_capture \
    --record_fps 15 \
    --record_every 1 \
    --record_queue_size 2 \
    --save_output \
    --output output \
    --show_preview

# ลดเวลาในการประมวลผล ลดการบันทึกวิดีโอ
    python sep_com_combined_steering_filter.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --enable_ros2 \
    --cmd_vel_publish_hz 30 \
    --fp16 \
    --async_capture \
    --record_fps 15 \
    --record_every 1 \
    --record_queue_size 2 \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 180 \
    --vehicle_height 280 \
    --lookahead_distance 150 \
    --target_lookahead_m 3.0 \
    --show_preview

    --save_output \
    --output output \
    --show_preview


# คำสั่งแนะนำสำหรับ real-time ไม่มีการบันทึกไฟล์:

ถ้าต้องการ latency ต่ำสุดจริง แนะนำใช้แบบนี้:

  python saty.py \
      --front_cam 0 \
      --left_cam 6 \
      --rear_cam 2 \
      --right_cam 4 \
      --dataset custom \
      --ckpt ./checkpoints_zoo/best27000.pth  \
      --enable_ros2 \
      --cmd_vel_publish_hz 10 \
      --fp16 \
      --async_capture \
      --model_input_width 640 \
      --model_input_height 360 \
      --vehicle_width 180 \
      --vehicle_height 280 \
      --lookahead_distance 150 \
      --target_lookahead_m 3.0 \
      --low_latency_mode

# เพิ่มการวิเคราะห์ความโค้งของเส้น spline พื่อส่งค่า linear x
    python saty3.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --enable_ros2 \
    --cmd_vel_publish_hz 20 \
    --fp16 \
    --async_capture \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 200 \
    --vehicle_height 300 \
    --lookahead_distance 150 \
    --target_lookahead_m 4.0 \
    --steering_filter_alpha 0.55 \
    --steering_deadband_deg 0.2 \
    --steering_max_rate_deg_s 60 \
    --spline_curvature_threshold 0.3 \
    --straight_linear_x 1.0 \
    --curved_linear_x 0.5 \
    --show_preview \
    --save_output \
    --output zoo22 \


    --low_latency_mode \


ถ้าต้องการบันทึกวิดีโอด้วย:
  python saty_timing.py \
      --front_cam 0 \
      --left_cam 4 \
      --rear_cam 2 \
      --right_cam 6 \
      --dataset custom \
      --ckpt ./zoo/iter_27000_deeplabv3plus_mobilenet_custom_os16.pth \
      --enable_ros2 \
      --cmd_vel_publish_hz 30 \
      --fp16 \
      --async_capture \
      --record_fps 15 \
      --record_every 1 \
      --record_queue_size 2 \
      --model_input_width 640 \
      --model_input_height 360 \
      --vehicle_width 190 \
      --vehicle_height 290 \
      --lookahead_distance 150 \
      --target_lookahead_m 3.0 \
      --save_output \
      --output output 


      --low_latency_mode

  

  

จาก get_argparser() ใน sep_com_combined.py ตอนนี้มีคำสั่ง/argument ดังนี้ครับ

  --front_cam INT

  ค่า default: 2
  กล้องหน้า

  --left_cam INT

  ค่า default: 4
  กล้องซ้าย

  --rear_cam INT

  ค่า default: 0
  กล้องหลัง

  --right_cam INT

  ค่า default: 6
  กล้องขวา

  --display_width INT
  --display_height INT

  ค่า default: 800, 600
  ขนาดหน้าต่างแสดงภาพกล้องรวม preview

  --enable_ros2

  เปิด ROS2 publisher สำหรับส่งคำสั่งควบคุม

  --cmd_vel_topic STR

  ค่า default: /cmd_vel
  topic ที่ publish Twist

  --dataset {voc,cityscapes,custom}

  ค่า default: cityscapes

  --output STR

  ค่า default: ./output
  โฟลเดอร์ output ถ้าเปิดบันทึกวิดีโอ

  --save_output

  เปิดบันทึกวิดีโอผลลัพธ์เป็น combined_nav_output.mp4

  --fps INT

  ค่า default: 15
  fps เป้าหมายของ loop และวิดีโอ output

  --show_preview

  เปิดหน้าต่าง preview ด้วย OpenCV

  --model MODEL_NAME

  ค่า default: deeplabv3plus_mobilenet
  ชื่อโมเดลจาก network.modeling

  --output_stride {8,16}

  ค่า default: 16

  --ckpt STR

  path checkpoint เช่น ./checkpoints/best2700.pth

  --gpu_id STR

  ค่า default: 0
  ตั้ง CUDA_VISIBLE_DEVICES

  --conf_thresh FLOAT

  ค่า default: 0.7
  threshold confidence สำหรับ drivable mask

  --lookahead_distance INT

  ค่า default: 200
  ระยะ offset/lookahead ในหน่วย pixel สำหรับเลือกจุด path

  --target_lookahead_m FLOAT

  ค่า default: 2.5
  ระยะ target point ของ pure pursuit ในหน่วยเมตร


  -------------------------------------------------------------------
  --cmd_vel_publish_hz 30
  ให้ node ส่ง /cmd_vel ซ้ำที่ 30 Hz หรือประมาณทุก 0.033 วินาที โดยใช้ค่าพวงมาลัยล่าสุด ช่วยให้เครื่องที่รัน
  steering_node.py ได้รับค่าถี่ขึ้น ไม่ต้องรอ inference เฟรมใหม่ทุกครั้ง

  --fp16
  ใช้ inference แบบ half precision บน CUDA เพื่อลดเวลาประมวลผลโมเดล ถ้า GPU รองรับจะเร็วขึ้น
  โดยยังใช้ขนาดภาพเดิม ไม่ได้ลด resolution

  --async_capture
  ให้ขั้นตอนรับภาพจากกล้องและสร้าง BEV ทำงานใน thread แยก แล้ว loop inference ใช้ “เฟรมล่าสุด”
  ช่วยลดการรอ capture/BEV ก่อนเริ่ม inference

  --save_output
  บันทึกวิดีโอ output เป็นไฟล์ เช่น combined_nav_output.mp4 และ combined_bev_output.mp4

  --record_fps 15
  กำหนด FPS ของวิดีโอที่บันทึกเป็น 15 FPS ไม่ได้หมายความว่า control loop ต้องวิ่ง 15 FPS แต่เป็น FPS
  ของไฟล์วิดีโอ output

  --record_every 1
  บันทึกทุกเฟรมที่ประมวลผลได้ ถ้าเปลี่ยนเป็น 2 จะบันทึก 1 เฟรมเว้น 1 เฟรม ช่วยลดภาระการเขียนไฟล์

  --record_queue_size 2
  ให้คิวบันทึกวิดีโอเก็บได้แค่ 2 เฟรม ถ้าเขียนไฟล์ไม่ทันจะทิ้งเฟรมเก่า เพื่อไม่ให้การบันทึกวิดีโอมาถ่วงการควบคุมรถ


# จดเวลาที่ใช้ในการประมวลผล

  python sep_com_combined_timing.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints/best2700.pth \
    --enable_ros2 \
    --cmd_vel_topic /cmd_vel \
    --show_preview \
    --timing_log combined_timing_marks.jsonl \
    --output output \
    --save_output

  ไฟล์ที่ได้จะมี 2 แบบ:

  timing_logs/combined_timing.csv
  output/combined_timing_marks.jsonl

  โดย timing_csv คือ CSV รายเฟรมที่ sep_com_combined_timing.py เขียนตรง ๆ ส่วน timing_log คือ raw timestamp ที่ใช้กับ
  timing_combined.py


  สร้าง report จาก timing log

  python3 timing_combined.py output/combined_timing_marks.jsonl

  ผลลัพธ์ default จะถูกเขียนในโฟลเดอร์เดียวกับ JSONL:

  output/timing_per_frame.csv
  output/timing_summary.csv
  output/timing_report.txt

  # กำหนดชื่อไฟล์ output เอง

  python3 timing_combined.py output/combined_timing_marks.jsonl \
    --per_frame_csv timing_logs/combined_per_frame.csv \
    --summary_csv timing_logs/combined_summary.csv \
    --report_txt timing_logs/combined_report.txt \
    --frame_report_every 30

  --frame_report_every 30 หมายถึงใน text report ให้แสดงรายละเอียดรายเฟรมทุก ๆ 30 แถว และจะใส่เฟรมสุดท้ายให้ด้วย

  ถ้าไม่อยากให้มี per-frame sample ใน report:

  python3 timing_combined.py output/combined_timing_marks.jsonl \
    --frame_report_every 0





















 # KR ZOO

python saty4.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --enable_ros2 \
    --cmd_vel_publish_hz 20 \
    --fp16 \
    --async_capture \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 200 \
    --vehicle_height 300 \
    --lookahead_distance 150 \
    --target_lookahead_m 4.0 \
    --steering_filter_alpha 0.55 \
    --steering_deadband_deg 0.2 \
    --steering_max_rate_deg_s 60 \
    --spline_curvature_threshold 0.3 \
    --straight_linear_x 1.5 \
    --curved_linear_x 1.0 \
    --show_preview \
    --save_output \
    --output zoo22 \
    


# แก้ไขให้ไปทางตรงมากกว่าจะไปทางแยก

python saty5.py \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --dataset custom \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --enable_ros2 \
    --cmd_vel_publish_hz 20 \
    --fp16 \
    --async_capture \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 200 \
    --vehicle_height 300 \
    --lookahead_distance 150 \
    --target_lookahead_m 4.0 \
    --steering_filter_alpha 0.55 \
    --steering_deadband_deg 0.2 \
    --steering_max_rate_deg_s 60 \
    --spline_curvature_threshold 0.3 \
    --straight_linear_x 1.0 \
    --curved_linear_x 0.5 \
    --show_preview \
    --save_output \
    --output zoo22 \
    --min_nav_path_points 50 \
    --straight_path_bias 1.5 \ #ใช้เพิ่มน้ำหนักให้ path อยู่ใกล้แนวกลางรถมากขึ้น ยิ่งค่าสูงยิ่งเลือกทางตรงมากขึ้น
    --nav_corridor_width 180 \ #ถ้าตั้งค่ามากกว่า 0 จะกรองเฉพาะพื้นที่ใน corridor กลางรถก่อนเลือก path
    --nav_path_offset_px -40 \ #ค่าลบ = ชิดซ้าย, ค่าบวก = ชิดขวา
    --nav_path_lateral_ratio 0.0 #offset ตามสัดส่วนความกว้างถนน -1.0 ถึง 1.0 ค่าลบ = ชิดซ้าย, ค่าบวก = ชิดขวา



python3 saty5.py \
      --front_cam 0 \
      --left_cam 2 \
      --rear_cam 4 \
      --right_cam 6 \
      --dataset custom \
      --ckpt ./checkpoints_zoo/best27000.pth \
      --enable_ros2 \
      --cmd_vel_publish_hz 20 \
      --fp16 \
      --async_capture \
      --model_input_width 640 \
      --model_input_height 360 \
      --vehicle_width 200 \
      --vehicle_height 300 \
      --lookahead_distance 150 \
      --target_lookahead_m 4.0 \
      --steering_filter_alpha 0.55 \
      --steering_deadband_deg 0.2 \
      --steering_max_rate_deg_s 60 \
      --spline_curvature_threshold 0.3 \
      --straight_linear_x 1.0 \
      --curved_linear_x 0.5 \
      --show_preview \
      --save_output \
      --output zoo22 \
      --min_nav_path_points 50 \



# เก็บค่าให้จบป โท เย่

ros2 bag record /cmd_vel -o bags/cmd_vel_$(date +%Y%m%d_%H%M%S)




# Zoo Detect with GPS

python3 zoo_detect.py \
    --input a.mp4 \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 200 \
    --vehicle_height 300 \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --dataset custom \
    --model deeplabv3plus_mobilenet \
    --show_preview \
    --cmd_vel_topic /cmd_vel_3




python zoo_detect.py \
    --use_camera \
    --front_cam 0 \
    --left_cam 4 \
    --rear_cam 2 \
    --right_cam 6 \
    --model deeplabv3plus_mobilenet \
    --dataset custom \
    --ckpt ./checkpoints_zoo/best27000.pth \
    --model_input_width 640 \
    --model_input_height 360 \
    --vehicle_width 200 \
    --vehicle_height 300 \
    --cmd_vel_topic /cmd_vel_3 \
    --show_preview