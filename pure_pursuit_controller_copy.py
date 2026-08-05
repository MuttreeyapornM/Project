#------------------------steering angle in rad unit--------------------------
#---------------------version 1 : speed = 2m/s-------------------------------
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PointStamped, Twist
# from std_msgs.msg import Float64
# import math
# import numpy as np

# class PurePursuitController(Node):
#     def __init__(self):
#         super().__init__('pure_pursuit_controller')
        
#         # Image/Pixel to Real World Conversion
#         self.pixel_to_meter = 0.01  # 1 pixel = 1 cm = 0.01 m
        
#         # Parameters - ปรับค่าเหล่านี้ให้เหมาะกับรถของคุณ (ในหน่วยเมตร)
#         self.wheelbase = 1.67  # ระยะห่างระหว่างเพลาหน้าและหลัง (เมตร)
#         self.lookahead_distance_pixels = 50  # 200 pixels = 2.0 เมตร
#         self.lookahead_distance = self.lookahead_distance_pixels * self.pixel_to_meter
#         self.max_steering_angle_deg = 50.0  # องศา (สำหรับการตั้งค่า)
#         self.max_steering_angle_rad = math.radians(self.max_steering_angle_deg)  # แปลงเป็นเรเดียน
#         self.target_speed = 2.0  # m/s (เริ่มต้นช้าๆ)
#         self.control_frequency = 20.0  # Hz
        
#         # Image coordinate system parameters
#         self.image_width = 1280   # ความกว้างภาพ (ปรับตามภาพจริง)
#         self.image_height = 720  # ความสูงภาพ (ปรับตามภาพจริง)
#         self.image_center_x = self.image_width // 2
#         self.image_center_y = self.image_height // 2
        
#         # Subscribers - รับจาก navigate_ros.py
#         self.destination_sub = self.create_subscription(
#             PointStamped, '/line1_destination_path', self.destination_callback, 10)
#         self.current_sub = self.create_subscription(
#             PointStamped, '/line2_current_path', self.current_callback, 10)

#         # Publishers สำหรับ drive-by-wire system
#         self.steering_pub = self.create_publisher(Float64, '/vehicle/steering_cmd', 10)
#         self.throttle_pub = self.create_publisher(Float64, '/vehicle/throttle_cmd', 10)
#         self.brake_pub = self.create_publisher(Float64, '/vehicle/brake_cmd', 10)
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
#         # State variables
#         self.line1_points = []  # จุดของ line1 (เส้นเป้าหมาย)
#         self.line2_points = []  # จุดของ line2 (เส้นปัจจุบัน)
#         self.current_position = None
#         self.line1_angle = 0.0  # มุมของ line1
#         self.line2_angle = 0.0  # มุมของ line2
#         self.angle_difference = 0.0  # ความแตกต่างของมุม
#         self.last_destination_time = None
#         self.last_current_time = None
        
#         # Control timer
#         self.control_timer = self.create_timer(
#             1.0/self.control_frequency, self.control_loop)
        
#         self.get_logger().info('Pure Pursuit Controller with Line Angle Calculation initialized')
#         self.get_logger().info(f'Max steering angle: {self.max_steering_angle_deg}° ({self.max_steering_angle_rad:.3f} rad)')
#         self.get_logger().info('Waiting for destination_path and current_path topics...')
        

#     def destination_callback(self, msg):
#         """รับจุดของ line1 (เส้นเป้าหมาย)"""
#         point_pixels = [msg.point.x, msg.point.y]
#         point_meters = self.pixels_to_meters(point_pixels)

#         self.get_logger().debug(f'[Line1 - destination_path] Received: {point_pixels} -> {point_meters}')
    
#         # เก็บจุดของ line1 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
#         if not self.line1_points or self.calculate_distance(point_meters, self.line1_points[-1]) > 0.01:
#             self.line1_points.append(point_meters)
#             if len(self.line1_points) > 5:  # เก็บ 5 จุดล่าสุด
#                 self.line1_points.pop(0)
        
#         # คำนวณมุมของ line1 และ line2
#         self.line1_angle = self.calculate_line_angle(self.line1_points)
        
#         # สำหรับ line2 อาจต้องการ heading angle แทน
#         # self.line2_heading = self.calculate_heading_angle(self.line2_points)
#         self.line1_angle = self.calculate_line_angle(self.line1_points)
#         self.last_destination_time = self.get_clock().now()

    
#     def current_callback(self, msg):
#         """รับจุดของ line2 (เส้นปัจจุบัน)"""
#         point_pixels = [msg.point.x, msg.point.y]
#         point_meters = self.pixels_to_meters(point_pixels)

#         self.get_logger().debug(f'[Line2 - current_path] Received: {point_pixels} -> {point_meters}')
    
#         # เก็บจุดของ line2 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
#         if not self.line2_points or self.calculate_distance(point_meters, self.line2_points[-1]) > 0.01:
#             self.line2_points.append(point_meters)
#             if len(self.line2_points) > 5:  # เก็บ 5 จุดล่าสุด
#                 self.line2_points.pop(0)
        
#         # คำนวณมุมของ line2 และตำแหน่งปัจจุบัน
#         self.line2_angle = self.calculate_line_angle(self.line2_points)
        
#         # อีกทางเลือก: ใช้ heading angle สำหรับ line2 (ทิศทางการเคลื่อนที่)
#         # self.line2_angle = self.calculate_heading_angle(self.line2_points)
#         self.current_position = point_meters  # ใช้จุดล่าสุดเป็นตำแหน่งปัจจุบัน
#         self.last_current_time = self.get_clock().now()

    
#     def pixels_to_meters(self, point_pixels):
#         """แปลงจากพิกเซลเป็นเมตร"""
#         x_meters = point_pixels[0] * self.pixel_to_meter
#         y_meters = point_pixels[1] * self.pixel_to_meter
#         return [x_meters, y_meters]
    
#     def meters_to_pixels(self, point_meters):
#         """แปลงจากเมตรเป็นพิกเซล (สำหรับ debug)"""
#         x_pixels = point_meters[0] / self.pixel_to_meter
#         y_pixels = point_meters[1] / self.pixel_to_meter
#         return [x_pixels, y_pixels]
    
#     def calculate_distance(self, point1, point2):
#         """คำนวณระยะทางระหว่าง 2 จุด"""
#         return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
#     def calculate_line_angle(self, points):
#         """
#         คำนวณมุม heading (ทิศทางการเคลื่อนที่) จากชุดจุด
        
#         Returns:
#             heading (float): มุม heading ในหน่วยเรเดียน
#                            - 0 rad = หันหน้าไปทางขวา (East) →
#                            - π/2 rad = หันหน้าไปทางบน (North) ↑
#                            - π rad = หันหน้าไปทางซ้าย (West) ←  
#                            - -π/2 rad = หันหน้าไปทางล่าง (South) ↓
#         """
#         if len(points) < 2:
#             return 0.0
            
#         # ใช้ทิศทางจากจุดก่อนหน้าไปจุดปัจจุบัน (ทิศทางการเคลื่อนที่)
#         if len(points) >= 2:
#             current_point = points[-1]  # จุดปัจจุบัน
#             previous_point = points[-2]  # จุดก่อนหน้า
            
#             dx = current_point[0] - previous_point[0]
#             dy = current_point[1] - previous_point[1]
            
#             # คำนวณทิศทางการเคลื่อนที่
#             heading = math.atan2(dy, dx)
#             return heading
        
#         return 0.0
#         """
#         คำนวณมุมของเส้นตรงจากชุดจุด (ใช้ linear regression)
        
#         Returns:
#             angle (float): มุมในหน่วยเรเดียน เทียบกับแกน X+ (แนวนอนไปทางขวา)
#                           - 0 rad (0°) = แนวนอนไปทางขวา →
#                           - π/2 rad (90°) = แนวตั้งไปทางบน ↑  
#                           - π rad (180°) = แนวนอนไปทางซ้าย ←
#                           - -π/2 rad (-90°) = แนวตั้งไปทางล่าง ↓
#         """
#         if len(points) < 2:
#             return 0.0
        
#         # ใช้ vector ระหว่างจุดแรกและจุดสุดท้าย (วิธีง่าย)
#         # หรือใช้ linear regression (วิธีแม่นยำ)
        
#         # วิธี 1: ใช้ vector ระหว่าง 2 จุดปลาย (เร็วและเข้าใจง่าย)
#         start_point = points[0]
#         end_point = points[-1]
        
#         dx = end_point[0] - start_point[0]
#         dy = end_point[1] - start_point[1]
        
#         # ใช้ atan2 เพื่อได้มุมที่ถูกต้องในทุก quadrant
#         angle = math.atan2(dy, dx)  # มุมเทียบกับแกน X+ 
        
#         return angle
        
#         # วิธี 2: Linear regression (สำหรับความแม่นยำมากขึ้น - เปิด comment ถ้าต้องการใช้)
#         """
#         x_coords = [p[0] for p in points]
#         y_coords = [p[1] for p in points]
        
#         # คำนวณค่าเฉลี่ย
#         x_mean = sum(x_coords) / len(x_coords)
#         y_mean = sum(y_coords) / len(y_coords)
        
#         # คำนวณความชัน (slope)
#         numerator = sum((x_coords[i] - x_mean) * (y_coords[i] - y_mean) for i in range(len(points)))
#         denominator = sum((x_coords[i] - x_mean) ** 2 for i in range(len(points)))
        
#         if abs(denominator) < 1e-10:  # หลีกเลี่ยงการหารด้วยศูนย์
#             # เส้นตั้งฉาก - ใช้ทิศทางจากจุดแรกไปจุดสุดท้าย
#             if len(points) >= 2:
#                 dy = points[-1][1] - points[0][1]
#                 return math.pi/2 if dy > 0 else -math.pi/2
#             return math.pi / 2
        
#         slope = numerator / denominator
#         angle = math.atan(slope)  # มุมในหน่วยเรเดียน เทียบกับแกน X+
        
#         return angle
#         """
    
#     def calculate_angle_based_steering(self):
#         """คำนวณมุมพวงมาลัยจากความแตกต่างของมุมระหว่าง line1 และ line2"""
#         if len(self.line1_points) < 2 or len(self.line2_points) < 2:
#             return 0.0
        
#         # คำนวณความแตกต่างของมุม
#         self.angle_difference = self.line1_angle - self.line2_angle
        
#         # Normalize angle ให้อยู่ในช่วง [-π, π]
#         self.angle_difference = math.atan2(math.sin(self.angle_difference), math.cos(self.angle_difference))
        
#         # แปลงความแตกต่างมุมเป็นมุมพวงมาลัย
#         # ใช้ proportional control กับ gain factor
#         steering_gain = 1.5  # ปรับค่านี้ตามความต้องการ
#         steering_angle_rad = self.angle_difference * steering_gain
        
#         # จำกัดมุมพวงมาลัย
#         steering_angle_rad = max(-self.max_steering_angle_rad, 
#                                 min(self.max_steering_angle_rad, steering_angle_rad))
        
#         return steering_angle_rad
    
#     def find_lookahead_point(self):
#         """หาจุด lookahead บน line1 (เส้นเป้าหมาย)"""
#         if not self.line1_points or not self.current_position:
#             return None
            
#         # หาจุดที่ใกล้ที่สุดบน line1
#         min_distance = float('inf')
#         closest_index = 0
        
#         for i, point in enumerate(self.line1_points):
#             distance = self.calculate_distance(point, self.current_position)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_index = i
        
#         # หาจุด lookahead โดยเริ่มจากจุดที่ใกล้ที่สุด
#         for i in range(closest_index, len(self.line1_points)):
#             point = self.line1_points[i]
#             distance = self.calculate_distance(point, self.current_position)
            
#             if distance >= self.lookahead_distance:
#                 return point
        
#         # ถ้าไม่เจอ ใช้จุดสุดท้าย
#         if self.line1_points:
#             return self.line1_points[-1]
        
#         return None
    
#     def calculate_steering_angle(self, lookahead_point):
#         """คำนวณมุมพวงมาลัยแบบผสมผสาน (Pure Pursuit + Angle Difference)"""
#         if not self.current_position:
#             return 0.0
        
#         # วิธีที่ 1: Pure Pursuit (ถ้ามี lookahead point)
#         pure_pursuit_angle = 0.0
#         if lookahead_point:
#             dx = lookahead_point[0] - self.current_position[0]
#             dy = lookahead_point[1] - self.current_position[1]
#             target_angle = math.atan2(dy, dx)
#             alpha = target_angle - self.line2_angle  # ใช้มุม line2 แทน current_heading
#             alpha = math.atan2(math.sin(alpha), math.cos(alpha))
#             ld = math.sqrt(dx**2 + dy**2)
            
#             if ld > 0.1:
#                 pure_pursuit_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
#         # วิธีที่ 2: Angle-based control
#         angle_based_steering = self.calculate_angle_based_steering()
        
#         # ผสมผสานทั้งสองวิธี
#         weight_pure_pursuit = 0.3  # น้ำหนักของ pure pursuit
#         weight_angle_based = 0.7   # น้ำหนักของ angle-based control
        
#         combined_steering = (weight_pure_pursuit * pure_pursuit_angle + 
#                            weight_angle_based * angle_based_steering)
        
#         # จำกัดมุมพวงมาลัย
#         combined_steering = max(-self.max_steering_angle_rad, 
#                               min(self.max_steering_angle_rad, combined_steering))
        
#         return combined_steering
    
#     def calculate_speed_control(self, lookahead_point):
#         """ควบคุมความเร็วแบบคงที่ และหยุดเมื่อไม่มี lookahead"""
#         if lookahead_point is None and len(self.line1_points) == 0:
#             # ไม่มีจุดถัดไป -> หยุดรถ
#             throttle = 0.0
#             brake = 1.0
#             target_speed = 0.0
#         else:
#             # ปรับความเร็วตามความแตกต่างของมุม
#             angle_diff_deg = abs(math.degrees(self.angle_difference))
            
#             if angle_diff_deg > 30:  # มุมแตกต่างมาก ลดความเร็ว
#                 speed_factor = 0.5
#             elif angle_diff_deg > 15:  # มุมแตกต่างปานกลาง ลดความเร็วเล็กน้อย
#                 speed_factor = 0.7
#             else:  # มุมใกล้เคียงกัน ความเร็วปกติ
#                 speed_factor = 1.0
            
#             throttle = 0.4 * speed_factor
#             brake = 0.0
#             target_speed = self.target_speed * speed_factor

#         return throttle, brake, target_speed

    
#     def control_loop(self):
#         """หลัก control loop"""
#         # ตรวจสอบว่ามีข้อมูลล่าสุดหรือไม่
#         current_time = self.get_clock().now()
        
#         if (self.last_destination_time is None or 
#             self.last_current_time is None):
#             self.get_logger().warn('Waiting for navigation data...')
#             return
        
#         # ตรวจสอบว่าข้อมูลยังใหม่อยู่หรือไม่ (ภายใน 1 วินาที)
#         time_diff_dest = (current_time - self.last_destination_time).nanoseconds / 1e9
#         time_diff_curr = (current_time - self.last_current_time).nanoseconds / 1e9
        
#         if time_diff_dest > 1.0 or time_diff_curr > 1.0:
#             self.get_logger().warn('Navigation data is stale, stopping vehicle')
#             self.publish_stop_command()
#             return
        
#         # หา lookahead point
#         lookahead_point = self.find_lookahead_point()
        
#         # คำนวณการควบคุม (ใช้ angle-based control เป็นหลัก)
#         steering_angle_rad = self.calculate_steering_angle(lookahead_point)
#         throttle, brake, target_speed = self.calculate_speed_control(lookahead_point)
        
#         # ส่งคำสั่งควบคุม
#         self.publish_control_commands(steering_angle_rad, throttle, brake, target_speed)
        
#         # Log information (ลดความถี่การ log)
#         if hasattr(self, '_log_counter'):
#             self._log_counter += 1
#         else:
#             self._log_counter = 0
            
#         if self._log_counter % 20 == 0:  # log ทุก 1 วินาที (20 Hz / 20)
#             line1_angle_deg = math.degrees(self.line1_angle)
#             line2_angle_deg = math.degrees(self.line2_angle)
#             angle_diff_deg = math.degrees(self.angle_difference)
#             steering_angle_deg = math.degrees(steering_angle_rad)
            
#             self.get_logger().info(
#                 f'Angles - Line1: {line1_angle_deg:.1f}°, Line2: {line2_angle_deg:.1f}°, '
#                 f'Diff: {angle_diff_deg:.1f}°, Steering: {steering_angle_deg:.1f}° ({steering_angle_rad:.3f} rad), '
#                 f'Speed: {target_speed:.1f} m/s'
#             )
    
#     def publish_control_commands(self, steering_angle_rad, throttle, brake, target_speed):
#         """ส่งคำสั่งควบคุมไปยัง drive-by-wire system"""
        
#         # Steering command - ส่งในหน่วยเรเดียน
#         steering_msg = Float64()
#         steering_msg.data = float(steering_angle_rad)  # เรเดียน
#         self.steering_pub.publish(steering_msg)
        
#         # Throttle command
#         throttle_msg = Float64()
#         throttle_msg.data = float(throttle)
#         self.throttle_pub.publish(throttle_msg)
        
#         # Brake command
#         brake_msg = Float64()
#         brake_msg.data = float(brake)
#         self.brake_pub.publish(brake_msg)
        
#         # Twist command (สำหรับ simulation)
#         cmd_vel = Twist()
#         cmd_vel.linear.x = float(target_speed)
#         if target_speed > 0:
#             cmd_vel.angular.z = float(steering_angle_rad * target_speed / self.wheelbase)
#         else:
#             cmd_vel.angular.z = 0.0
#         self.cmd_vel_pub.publish(cmd_vel)
    
#     def publish_stop_command(self):
#         """ส่งคำสั่งหยุดรถ"""
#         self.publish_control_commands(0.0, 0.0, 0.5, 0.0)

# def main(args=None):
#     rclpy.init(args=args)
    
#     controller = PurePursuitController()
    
#     try:
#         rclpy.spin(controller)
#     except KeyboardInterrupt:
#         controller.get_logger().info('Shutting down Pure Pursuit Controller')
#     finally:
#         controller.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



#------------version2 fiix speedd = 1m/s and 0° = แนวตั้งขึ้น ↑ 90° = แนวนอนขวา → 180° = แนวตั้งลง ↓-90° = แนวนอนซ้าย ←--------
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PointStamped, Twist
# from std_msgs.msg import Float64
# import math
# import numpy as np

# class PurePursuitController(Node):
#     def __init__(self):
#         super().__init__('pure_pursuit_controller')
        
#         # Image/Pixel to Real World Conversion
#         self.pixel_to_meter = 0.01  # 1 pixel = 1 cm = 0.01 m
        
#         # Parameters - ปรับค่าเหล่านี้ให้เหมาะกับรถของคุณ (ในหน่วยเมตร)
#         self.wheelbase = 1.67  # ระยะห่างระหว่างเพลาหน้าและหลัง (เมตร)
#         self.lookahead_distance_pixels = 50  # 200 pixels = 2.0 เมตร
#         self.lookahead_distance = self.lookahead_distance_pixels * self.pixel_to_meter
#         self.max_steering_angle_deg = 50.0  # องศา (สำหรับการตั้งค่า)
#         self.max_steering_angle_rad = math.radians(self.max_steering_angle_deg)  # แปลงเป็นเรเดียน
#         self.target_speed = 1.0  # m/s (Fixed speed value)
#         self.control_frequency = 20.0  # Hz
        
#         # Image coordinate system parameters
#         self.image_width = 1280   # ความกว้างภาพ (ปรับตามภาพจริง)
#         self.image_height = 720  # ความสูงภาพ (ปรับตามภาพจริง)
#         self.image_center_x = self.image_width // 2
#         self.image_center_y = self.image_height // 2
        
#         # Subscribers - รับจาก navigate_ros.py
#         self.destination_sub = self.create_subscription(
#             PointStamped, '/line1_destination_path', self.destination_callback, 10)
#         self.current_sub = self.create_subscription(
#             PointStamped, '/line2_current_path', self.current_callback, 10)

#         # Publishers สำหรับ drive-by-wire system
#         self.steering_pub = self.create_publisher(Float64, '/vehicle/steering_cmd', 10)
#         self.throttle_pub = self.create_publisher(Float64, '/vehicle/throttle_cmd', 10)
#         self.brake_pub = self.create_publisher(Float64, '/vehicle/brake_cmd', 10)
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
#         # State variables
#         self.line1_points = []  # จุดของ line1 (เส้นเป้าหมาย)
#         self.line2_points = []  # จุดของ line2 (เส้นปัจจุบัน)
#         self.current_position = None
#         self.line1_angle = 0.0  # มุมของ line1
#         self.line2_angle = 0.0  # มุมของ line2
#         self.angle_difference = 0.0  # ความแตกต่างของมุม
#         self.last_destination_time = None
#         self.last_current_time = None
        
#         # Control timer
#         self.control_timer = self.create_timer(
#             1.0/self.control_frequency, self.control_loop)
        
#         self.get_logger().info('Pure Pursuit Controller with Line Angle Calculation initialized')
#         self.get_logger().info(f'Max steering angle: {self.max_steering_angle_deg}° ({self.max_steering_angle_rad:.3f} rad)')
#         self.get_logger().info(f'Fixed target speed: {self.target_speed} m/s')
#         self.get_logger().info('Waiting for destination_path and current_path topics...')
        

#     def destination_callback(self, msg):
#         """รับจุดของ line1 (เส้นเป้าหมาย)"""
#         point_pixels = [msg.point.x, msg.point.y]
#         point_meters = self.pixels_to_meters(point_pixels)

#         self.get_logger().debug(f'[Line1 - destination_path] Received: {point_pixels} -> {point_meters}')
    
#         # เก็บจุดของ line1 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
#         if not self.line1_points or self.calculate_distance(point_meters, self.line1_points[-1]) > 0.01:
#             self.line1_points.append(point_meters)
#             if len(self.line1_points) > 5:  # เก็บ 5 จุดล่าสุด
#                 self.line1_points.pop(0)
        
#         # คำนวณมุมของ line1 และ line2
#         self.line1_angle = self.calculate_line_angle(self.line1_points)
        
#         # สำหรับ line2 อาจต้องการ heading angle แทน
#         # self.line2_heading = self.calculate_heading_angle(self.line2_points)
#         self.line1_angle = self.calculate_line_angle(self.line1_points)
#         self.last_destination_time = self.get_clock().now()

    
#     def current_callback(self, msg):
#         """รับจุดของ line2 (เส้นปัจจุบัน)"""
#         point_pixels = [msg.point.x, msg.point.y]
#         point_meters = self.pixels_to_meters(point_pixels)

#         self.get_logger().debug(f'[Line2 - current_path] Received: {point_pixels} -> {point_meters}')
    
#         # เก็บจุดของ line2 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
#         if not self.line2_points or self.calculate_distance(point_meters, self.line2_points[-1]) > 0.01:
#             self.line2_points.append(point_meters)
#             if len(self.line2_points) > 5:  # เก็บ 5 จุดล่าสุด
#                 self.line2_points.pop(0)
        
#         # คำนวณมุมของ line2 และตำแหน่งปัจจุบัน
#         self.line2_angle = self.calculate_line_angle(self.line2_points)
        
#         # อีกทางเลือก: ใช้ heading angle สำหรับ line2 (ทิศทางการเคลื่อนที่)
#         # self.line2_angle = self.calculate_heading_angle(self.line2_points)
#         self.current_position = point_meters  # ใช้จุดล่าสุดเป็นตำแหน่งปัจจุบัน
#         self.last_current_time = self.get_clock().now()

    
#     def pixels_to_meters(self, point_pixels):
#         """แปลงจากพิกเซลเป็นเมตร"""
#         x_meters = point_pixels[0] * self.pixel_to_meter
#         y_meters = point_pixels[1] * self.pixel_to_meter
#         return [x_meters, y_meters]
    
#     def meters_to_pixels(self, point_meters):
#         """แปลงจากเมตรเป็นพิกเซล (สำหรับ debug)"""
#         x_pixels = point_meters[0] / self.pixel_to_meter
#         y_pixels = point_meters[1] / self.pixel_to_meter
#         return [x_pixels, y_pixels]
    
#     def calculate_distance(self, point1, point2):
#         """คำนวณระยะทางระหว่าง 2 จุด"""
#         return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
#     def calculate_line_angle(self, points):
#         """
#         คำนวณมุม heading (ทิศทางการเคลื่อนที่) จากชุดจุด
#         Returns:
#         heading (float): มุม heading ในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
#         - 0 rad (0°) = แนวตั้งไปทางบน ↑
#         - π/2 rad (90°) = แนวนอนไปทางขวา →
#         - π rad (180°) = แนวตั้งไปทางล่าง ↓
#         - -π/2 rad (-90°) = แนวนอนไปทางซ้าย ←
#         """
#         if len(points) < 2:
#             return 0.0
    
#         # ใช้ทิศทางจากจุดก่อนหน้าไปจุดปัจจุบัน (ทิศทางการเคลื่อนที่)
#         if len(points) >= 2:
#             current_point = points[-1]  # จุดปัจจุบัน
#             previous_point = points[-2]  # จุดก่อนหน้า
#             dx = current_point[0] - previous_point[0]
#             dy = current_point[1] - previous_point[1]
        
#             # คำนวณทิศทางการเคลื่อนที่เทียบกับแกน Y+
#             # ใช้ atan2(dx, dy) แทน atan2(dy, dx) เพื่อให้เทียบกับแกน Y+
#             heading = math.atan2(dx, dy)
#             return heading
    
#         return 0.0


#     def calculate_line_angle_from_points(self, points):
#         """
#         คำนวณมุมของเส้นตรงจากชุดจุด (ใช้ linear regression)
#         Returns:
#         angle (float): มุมในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
#         - 0 rad (0°) = แนวตั้งไปทางบน ↑
#         - π/2 rad (90°) = แนวนอนไปทางขวา →
#         - π rad (180°) = แนวตั้งไปทางล่าง ↓
#         - -π/2 rad (-90°) = แนวนอนไปทางซ้าย ←
#         """
#         if len(points) < 2:
#             return 0.0
    
#         # วิธี 1: ใช้ vector ระหว่าง 2 จุดปลาย (เร็วและเข้าใจง่าย)
#         start_point = points[0]
#         end_point = points[-1]
#         dx = end_point[0] - start_point[0]
#         dy = end_point[1] - start_point[1]
    
#         # ใช้ atan2(dx, dy) เพื่อได้มุมเทียบกับแกน Y+ (แนวตั้งไปทางบน)
#         angle = math.atan2(dx, dy)
#         return angle


#     def calculate_line_angle_linear_regression(self, points):
#         """
#         คำนวณมุมของเส้นตรงจากชุดจุดโดยใช้ linear regression (แม่นยำกว่า)
#         Returns:
#         angle (float): มุมในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
#         """
#         if len(points) < 2:
#             return 0.0
    
#         # แยก x และ y coordinates
#         x_coords = [point[0] for point in points]
#         y_coords = [point[1] for point in points]
    
#         # คำนวณ linear regression
#         n = len(points)
#         sum_x = sum(x_coords)
#         sum_y = sum(y_coords)
#         sum_xy = sum(x * y for x, y in zip(x_coords, y_coords))
#         sum_x2 = sum(x * x for x in x_coords)
    
#         # คำนวณ slope (m)
#         denominator = n * sum_x2 - sum_x * sum_x
#         if abs(denominator) < 1e-10:  # หลีกเลี่ยงการหารด้วย 0
#             return 0.0
    
#         slope = (n * sum_xy - sum_x * sum_y) / denominator
    
#         # แปลง slope เป็นมุมเทียบกับแกน Y+
#         # slope = dy/dx, แต่เราต้องการมุมเทียบกับแกน Y+
#         # ดังนั้นใช้ atan(1/slope) หรือ atan2(1, slope)
#         if abs(slope) < 1e-10:  # slope ≈ 0 (เส้นแนวนอน)
#             return math.pi / 2  # 90 degrees (แนวนอน)
#         else:
#             # ใช้ atan2 เพื่อจัดการ quadrant ที่ถูกต้อง
#             angle = math.atan2(1, slope)
#             return angle
    
#     def calculate_angle_based_steering(self):
#         """คำนวณมุมพวงมาลัยจากความแตกต่างของมุมระหว่าง line1 และ line2"""
#         if len(self.line1_points) < 2 or len(self.line2_points) < 2:
#             return 0.0
        
#         # คำนวณความแตกต่างของมุม
#         self.angle_difference = self.line1_angle - self.line2_angle
        
#         # Normalize angle ให้อยู่ในช่วง [-π, π]
#         self.angle_difference = math.atan2(math.sin(self.angle_difference), math.cos(self.angle_difference))
        
#         # แปลงความแตกต่างมุมเป็นมุมพวงมาลัย
#         # ใช้ proportional control กับ gain factor
#         steering_gain = 1.5  # ปรับค่านี้ตามความต้องการ
#         steering_angle_rad = self.angle_difference * steering_gain
        
#         # จำกัดมุมพวงมาลัย
#         steering_angle_rad = max(-self.max_steering_angle_rad, 
#                                 min(self.max_steering_angle_rad, steering_angle_rad))
        
#         return steering_angle_rad
    
#     def find_lookahead_point(self):
#         """หาจุด lookahead บน line1 (เส้นเป้าหมาย)"""
#         if not self.line1_points or not self.current_position:
#             return None
            
#         # หาจุดที่ใกล้ที่สุดบน line1
#         min_distance = float('inf')
#         closest_index = 0
        
#         for i, point in enumerate(self.line1_points):
#             distance = self.calculate_distance(point, self.current_position)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_index = i
        
#         # หาจุด lookahead โดยเริ่มจากจุดที่ใกล้ที่สุด
#         for i in range(closest_index, len(self.line1_points)):
#             point = self.line1_points[i]
#             distance = self.calculate_distance(point, self.current_position)
            
#             if distance >= self.lookahead_distance:
#                 return point
        
#         # ถ้าไม่เจอ ใช้จุดสุดท้าย
#         if self.line1_points:
#             return self.line1_points[-1]
        
#         return None
    
#     def calculate_steering_angle(self, lookahead_point):
#         """คำนวณมุมพวงมาลัยแบบผสมผสาน (Pure Pursuit + Angle Difference)"""
#         if not self.current_position:
#             return 0.0
        
#         # วิธีที่ 1: Pure Pursuit (ถ้ามี lookahead point)
#         pure_pursuit_angle = 0.0
#         if lookahead_point:
#             dx = lookahead_point[0] - self.current_position[0]
#             dy = lookahead_point[1] - self.current_position[1]
#             target_angle = math.atan2(dy, dx)
#             alpha = target_angle - self.line2_angle  # ใช้มุม line2 แทน current_heading
#             alpha = math.atan2(math.sin(alpha), math.cos(alpha))
#             ld = math.sqrt(dx**2 + dy**2)
            
#             if ld > 0.1:
#                 pure_pursuit_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
#         # วิธีที่ 2: Angle-based control
#         angle_based_steering = self.calculate_angle_based_steering()
        
#         # ผสมผสานทั้งสองวิธี
#         weight_pure_pursuit = 0.3  # น้ำหนักของ pure pursuit
#         weight_angle_based = 0.7   # น้ำหนักของ angle-based control
        
#         combined_steering = (weight_pure_pursuit * pure_pursuit_angle + 
#                            weight_angle_based * angle_based_steering)
        
#         # จำกัดมุมพวงมาลัย
#         combined_steering = max(-self.max_steering_angle_rad, 
#                               min(self.max_steering_angle_rad, combined_steering))
        
#         return combined_steering
    
#     def calculate_speed_control(self, lookahead_point):
#         """ควบคุมความเร็วแบบคงที่ที่ 1.0 m/s"""
#         if lookahead_point is None and len(self.line1_points) == 0:
#             # ไม่มีจุดถัดไป -> หยุดรถ
#             throttle = 0.0
#             brake = 1.0
#             target_speed = 0.0
#         else:
#             # ใช้ความเร็วคงที่ที่ 1.0 m/s
#             throttle = 0.4  # ค่าคงที่สำหรับการเร่ง
#             brake = 0.0
#             target_speed = self.target_speed  # 1.0 m/s

#         return throttle, brake, target_speed

    
#     def control_loop(self):
#         """หลัก control loop"""
#         # ตรวจสอบว่ามีข้อมูลล่าสุดหรือไม่
#         current_time = self.get_clock().now()
        
#         if (self.last_destination_time is None or 
#             self.last_current_time is None):
#             self.get_logger().warn('Waiting for navigation data...')
#             return
        
#         # ตรวจสอบว่าข้อมูลยังใหม่อยู่หรือไม่ (ภายใน 1 วินาที)
#         time_diff_dest = (current_time - self.last_destination_time).nanoseconds / 1e9
#         time_diff_curr = (current_time - self.last_current_time).nanoseconds / 1e9
        
#         if time_diff_dest > 1.0 or time_diff_curr > 1.0:
#             self.get_logger().warn('Navigation data is stale, stopping vehicle')
#             self.publish_stop_command()
#             return
        
#         # หา lookahead point
#         lookahead_point = self.find_lookahead_point()
        
#         # คำนวณการควบคุม (ใช้ angle-based control เป็นหลัก)
#         steering_angle_rad = self.calculate_steering_angle(lookahead_point)
#         throttle, brake, target_speed = self.calculate_speed_control(lookahead_point)
        
#         # ส่งคำสั่งควบคุม
#         self.publish_control_commands(steering_angle_rad, throttle, brake, target_speed)
        
#         # Log information (ลดความถี่การ log)
#         if hasattr(self, '_log_counter'):
#             self._log_counter += 1
#         else:
#             self._log_counter = 0
            
#         if self._log_counter % 20 == 0:  # log ทุก 1 วินาที (20 Hz / 20)
#             line1_angle_deg = math.degrees(self.line1_angle)
#             line2_angle_deg = math.degrees(self.line2_angle)
#             angle_diff_deg = math.degrees(self.angle_difference)
#             steering_angle_deg = math.degrees(steering_angle_rad)
            
#             self.get_logger().info(
#                 f'Angles - Line1: {line1_angle_deg:.1f}°, Line2: {line2_angle_deg:.1f}°, '
#                 f'Diff: {angle_diff_deg:.1f}°, Steering: {steering_angle_deg:.1f}° ({steering_angle_rad:.3f} rad), '
#                 f'Speed: {target_speed:.1f} m/s (Fixed)'
#             )
    
#     def publish_control_commands(self, steering_angle_rad, throttle, brake, target_speed):
#         """ส่งคำสั่งควบคุมไปยัง drive-by-wire system"""
        
#         # Steering command - ส่งในหน่วยเรเดียน
#         steering_msg = Float64()
#         steering_msg.data = float(steering_angle_rad)  # เรเดียน
#         self.steering_pub.publish(steering_msg)
        
#         # Throttle command
#         throttle_msg = Float64()
#         throttle_msg.data = float(throttle)
#         self.throttle_pub.publish(throttle_msg)
        
#         # Brake command
#         brake_msg = Float64()
#         brake_msg.data = float(brake)
#         self.brake_pub.publish(brake_msg)
        
#         # Twist command (สำหรับ simulation)
#         cmd_vel = Twist()
#         cmd_vel.linear.x = float(target_speed)
#         if target_speed > 0:
#             cmd_vel.angular.z = float(steering_angle_rad * target_speed / self.wheelbase)
#         else:
#             cmd_vel.angular.z = 0.0
#         self.cmd_vel_pub.publish(cmd_vel)
    
#     def publish_stop_command(self):
#         """ส่งคำสั่งหยุดรถ"""
#         self.publish_control_commands(0.0, 0.0, 0.5, 0.0)

# def main(args=None):
#     rclpy.init(args=args)
    
#     controller = PurePursuitController()
    
#     try:
#         rclpy.spin(controller)
#     except KeyboardInterrupt:
#         controller.get_logger().info('Shutting down Pure Pursuit Controller')
#     finally:
#         controller.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()





#------------version3 fiix speedd = 1m/s and - 0 rad (0°) = แนวตั้งไปทางบน ↑ - π/2 rad (90°) = แนวนอนไปทางขวา → - π rad (180°) = แนวตั้งไปทางล่าง ↓ -π/2 rad (-90°) = แนวนอนไปทางซ้าย ←
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Float64
import math
import numpy as np

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')
        
        # Image/Pixel to Real World Conversion
        self.pixel_to_meter = 0.01  # 1 pixel = 1 cm = 0.01 m
        
        # Parameters - ปรับค่าเหล่านี้ให้เหมาะกับรถของคุณ (ในหน่วยเมตร)
        self.wheelbase = 1.67  # ระยะห่างระหว่างเพลาหน้าและหลัง (เมตร)
        self.lookahead_distance_pixels = 50  # 200 pixels = 2.0 เมตร
        self.lookahead_distance = self.lookahead_distance_pixels * self.pixel_to_meter
        self.max_steering_angle_deg = 50.0  # องศา (สำหรับการตั้งค่า)
        self.max_steering_angle_rad = math.radians(self.max_steering_angle_deg)  # แปลงเป็นเรเดียน
        self.target_speed = 1.0  # m/s (Fixed speed value)
        self.control_frequency = 20.0  # Hz
        
        # Image coordinate system parameters
        self.image_width = 1280   # ความกว้างภาพ (ปรับตามภาพจริง)
        self.image_height = 720  # ความสูงภาพ (ปรับตามภาพจริง)
        self.image_center_x = self.image_width // 2
        self.image_center_y = self.image_height // 2
        
        # Subscribers - รับจาก navigate_ros.py
        self.destination_sub = self.create_subscription(
            PointStamped, '/line1_destination_path', self.destination_callback, 10)
        self.current_sub = self.create_subscription(
            PointStamped, '/line2_current_path', self.current_callback, 10)

        # Publishers สำหรับ drive-by-wire system
        self.steering_pub = self.create_publisher(Float64, '/vehicle/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, '/vehicle/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/vehicle/brake_cmd', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State variables
        self.line1_points = []  # จุดของ line1 (เส้นเป้าหมาย)
        self.line2_points = []  # จุดของ line2 (เส้นปัจจุบัน)
        self.current_position = None
        self.line1_angle = 0.0  # มุมของ line1
        self.line2_angle = 0.0  # มุมของ line2
        self.angle_difference = 0.0  # ความแตกต่างของมุม
        self.last_destination_time = None
        self.last_current_time = None
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0/self.control_frequency, self.control_loop)
        
        self.get_logger().info('Pure Pursuit Controller with Line Angle Calculation initialized')
        self.get_logger().info(f'Max steering angle: {self.max_steering_angle_deg}° ({self.max_steering_angle_rad:.3f} rad)')
        self.get_logger().info(f'Fixed target speed: {self.target_speed} m/s')
        self.get_logger().info('Waiting for destination_path and current_path topics...')
        

    def destination_callback(self, msg):
        """รับจุดของ line1 (เส้นเป้าหมาย)"""
        point_pixels = [msg.point.x, msg.point.y]
        point_meters = self.pixels_to_meters(point_pixels)

        self.get_logger().debug(f'[Line1 - destination_path] Received: {point_pixels} -> {point_meters}')
    
        # เก็บจุดของ line1 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
        if not self.line1_points or self.calculate_distance(point_meters, self.line1_points[-1]) > 0.01:
            self.line1_points.append(point_meters)
            if len(self.line1_points) > 5:  # เก็บ 5 จุดล่าสุด
                self.line1_points.pop(0)
        
        # คำนวณมุมของ line1 และ line2
        self.line1_angle = self.calculate_line_angle(self.line1_points)
        
        # สำหรับ line2 อาจต้องการ heading angle แทน
        # self.line2_heading = self.calculate_heading_angle(self.line2_points)
        self.line1_angle = self.calculate_line_angle(self.line1_points)
        self.last_destination_time = self.get_clock().now()

    
    def current_callback(self, msg):
        """รับจุดของ line2 (เส้นปัจจุบัน)"""
        point_pixels = [msg.point.x, msg.point.y]
        point_meters = self.pixels_to_meters(point_pixels)

        self.get_logger().debug(f'[Line2 - current_path] Received: {point_pixels} -> {point_meters}')
    
        # เก็บจุดของ line2 (เก็บมากกว่า 1 จุดเพื่อคำนวณมุม)
        if not self.line2_points or self.calculate_distance(point_meters, self.line2_points[-1]) > 0.01:
            self.line2_points.append(point_meters)
            if len(self.line2_points) > 5:  # เก็บ 5 จุดล่าสุด
                self.line2_points.pop(0)
        
        # คำนวณมุมของ line2 และตำแหน่งปัจจุบัน
        self.line2_angle = self.calculate_line_angle(self.line2_points)
        
        # อีกทางเลือก: ใช้ heading angle สำหรับ line2 (ทิศทางการเคลื่อนที่)
        # self.line2_angle = self.calculate_heading_angle(self.line2_points)
        self.current_position = point_meters  # ใช้จุดล่าสุดเป็นตำแหน่งปัจจุบัน
        self.last_current_time = self.get_clock().now()

    
    def pixels_to_meters(self, point_pixels):
        """แปลงจากพิกเซลเป็นเมตร"""
        x_meters = point_pixels[0] * self.pixel_to_meter
        y_meters = point_pixels[1] * self.pixel_to_meter
        return [x_meters, y_meters]
    
    def meters_to_pixels(self, point_meters):
        """แปลงจากเมตรเป็นพิกเซล (สำหรับ debug)"""
        x_pixels = point_meters[0] / self.pixel_to_meter
        y_pixels = point_meters[1] / self.pixel_to_meter
        return [x_pixels, y_pixels]
    
    def calculate_distance(self, point1, point2):
        """คำนวณระยะทางระหว่าง 2 จุด"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_line_angle(self, points):
        """
        คำนวณมุม heading (ทิศทางการเคลื่อนที่) จากชุดจุด
        Returns:
        heading (float): มุม heading ในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
        - 0 rad (0°) = แนวตั้งไปทางบน ↑
        - π/2 rad (90°) = แนวนอนไปทางขวา →
        - π rad (180°) = แนวตั้งไปทางล่าง ↓
        - -π/2 rad (-90°) = แนวนอนไปทางซ้าย ←
        """
        if len(points) < 2:
            return 0.0
    
        # ใช้ทิศทางจากจุดก่อนหน้าไปจุดปัจจุบัน (ทิศทางการเคลื่อนที่)
        if len(points) >= 2:
            current_point = points[-1]  # จุดปัจจุบัน
            previous_point = points[-2]  # จุดก่อนหน้า
            dx = current_point[0] - previous_point[0]
            dy = current_point[1] - previous_point[1]
        
            # คำนวณทิศทางการเคลื่อนที่เทียบกับแกน Y+
            # สูตร: atan2(-dx, dy) เพื่อให้ได้มุมที่ถูกต้องตามต้องการ
            # - เมื่อ dy > 0, dx = 0 → atan2(0, positive) = 0 → ทิศเหนือ
            # - เมื่อ dy = 0, dx > 0 → atan2(-positive, 0) = -π/2 → ทิศตะวันออก
            # - เมื่อ dy < 0, dx = 0 → atan2(0, negative) = π → ทิศใต้  
            # - เมื่อ dy = 0, dx < 0 → atan2(positive, 0) = π/2 → ทิศตะวันตก
            heading = math.atan2(-dx, dy)
            return heading
    
        return 0.0


    def calculate_line_angle_from_points(self, points):
        """
        คำนวณมุมของเส้นตรงจากชุดจุด (ใช้ linear regression)
        Returns:
        angle (float): มุมในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
        - 0 rad (0°) = แนวตั้งไปทางบน ↑
        - π/2 rad (90°) = แนวนอนไปทางขวา →
        - π rad (180°) = แนวตั้งไปทางล่าง ↓
        - -π/2 rad (-90°) = แนวนอนไปทางซ้าย ←
        """
        if len(points) < 2:
            return 0.0
    
        # วิธี 1: ใช้ vector ระหว่าง 2 จุดปลาย (เร็วและเข้าใจง่าย)
        start_point = points[0]
        end_point = points[-1]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
    
        # ใช้ atan2(-dx, dy) เพื่อได้มุมเทียบกับแกน Y+ ตามที่ต้องการ
        angle = math.atan2(-dx, dy)
        return angle


    def calculate_line_angle_linear_regression(self, points):
        """
        คำนวณมุมของเส้นตรงจากชุดจุดโดยใช้ linear regression (แม่นยำกว่า)
        Returns:
        angle (float): มุมในหน่วยเรเดียน เทียบกับแกน Y+ (แนวตั้งไปทางบน)
        """
        if len(points) < 2:
            return 0.0
    
        # แยก x และ y coordinates
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
    
        # คำนวณ linear regression
        n = len(points)
        sum_x = sum(x_coords)
        sum_y = sum(y_coords)
        sum_xy = sum(x * y for x, y in zip(x_coords, y_coords))
        sum_x2 = sum(x * x for x in x_coords)
    
        # คำนวณ slope (m)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:  # หลีกเลี่ยงการหารด้วย 0
            return 0.0
    
        slope = (n * sum_xy - sum_x * sum_y) / denominator
    
        # แปลง slope เป็นมุมเทียบกับแกน Y+
        # slope = dy/dx สำหรับแกน X+, แต่เราต้องการเทียบกับแกน Y+
        # ใช้ atan2(-1, slope) เพื่อแปลงเป็นมุมที่ถูกต้อง
        if abs(slope) < 1e-10:  # slope ≈ 0 (เส้นแนวนอน)
            return -math.pi / 2  # -90 degrees (แนวนอนไปทางขวา)
        else:
            # ใช้ atan2 เพื่อจัดการ quadrant ที่ถูกต้อง
            angle = math.atan2(-1, slope)
            return angle
            
    def calculate_angle_based_steering(self):
        """คำนวณมุมพวงมาลัยจากความแตกต่างของมุมระหว่าง line1 และ line2"""
        if len(self.line1_points) < 2 or len(self.line2_points) < 2:
            return 0.0
        
        # คำนวณความแตกต่างของมุม
        self.angle_difference = self.line1_angle - self.line2_angle
        
        # Normalize angle ให้อยู่ในช่วง [-π, π]
        self.angle_difference = math.atan2(math.sin(self.angle_difference), math.cos(self.angle_difference))
        
        # แปลงความแตกต่างมุมเป็นมุมพวงมาลัย
        # ใช้ proportional control กับ gain factor
        steering_gain = 1.5  # ปรับค่านี้ตามความต้องการ
        steering_angle_rad = self.angle_difference * steering_gain
        
        # จำกัดมุมพวงมาลัย
        steering_angle_rad = max(-self.max_steering_angle_rad, 
                                min(self.max_steering_angle_rad, steering_angle_rad))
        
        return steering_angle_rad
    
    def find_lookahead_point(self):
        """หาจุด lookahead บน line1 (เส้นเป้าหมาย)"""
        if not self.line1_points or not self.current_position:
            return None
            
        # หาจุดที่ใกล้ที่สุดบน line1
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.line1_points):
            distance = self.calculate_distance(point, self.current_position)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # หาจุด lookahead โดยเริ่มจากจุดที่ใกล้ที่สุด
        for i in range(closest_index, len(self.line1_points)):
            point = self.line1_points[i]
            distance = self.calculate_distance(point, self.current_position)
            
            if distance >= self.lookahead_distance:
                return point
        
        # ถ้าไม่เจอ ใช้จุดสุดท้าย
        if self.line1_points:
            return self.line1_points[-1]
        
        return None
    
    def calculate_steering_angle(self, lookahead_point):
        """คำนวณมุมพวงมาลัยแบบผสมผสาน (Pure Pursuit + Angle Difference)"""
        if not self.current_position:
            return 0.0
        
        # วิธีที่ 1: Pure Pursuit (ถ้ามี lookahead point)
        pure_pursuit_angle = 0.0
        if lookahead_point:
            dx = lookahead_point[0] - self.current_position[0]
            dy = lookahead_point[1] - self.current_position[1]
            target_angle = math.atan2(dy, dx)
            alpha = target_angle - self.line2_angle  # ใช้มุม line2 แทน current_heading
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            ld = math.sqrt(dx**2 + dy**2)
            
            if ld > 0.1:
                pure_pursuit_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
        # วิธีที่ 2: Angle-based control
        angle_based_steering = self.calculate_angle_based_steering()
        
        # ผสมผสานทั้งสองวิธี
        weight_pure_pursuit = 0.3  # น้ำหนักของ pure pursuit
        weight_angle_based = 0.7   # น้ำหนักของ angle-based control
        
        combined_steering = (weight_pure_pursuit * pure_pursuit_angle + 
                           weight_angle_based * angle_based_steering)
        
        # จำกัดมุมพวงมาลัย
        combined_steering = max(-self.max_steering_angle_rad, 
                              min(self.max_steering_angle_rad, combined_steering))
        
        return combined_steering
    
    def calculate_speed_control(self, lookahead_point):
        """ควบคุมความเร็วแบบคงที่ที่ 1.0 m/s"""
        if lookahead_point is None and len(self.line1_points) == 0:
            # ไม่มีจุดถัดไป -> หยุดรถ
            throttle = 0.0
            brake = 1.0
            target_speed = 0.0
        else:
            # ใช้ความเร็วคงที่ที่ 1.0 m/s
            throttle = 0.4  # ค่าคงที่สำหรับการเร่ง
            brake = 0.0
            target_speed = self.target_speed  # 1.0 m/s

        return throttle, brake, target_speed

    
    def control_loop(self):
        """หลัก control loop"""
        # ตรวจสอบว่ามีข้อมูลล่าสุดหรือไม่
        current_time = self.get_clock().now()
        
        if (self.last_destination_time is None or 
            self.last_current_time is None):
            self.get_logger().warn('Waiting for navigation data...')
            return
        
        # ตรวจสอบว่าข้อมูลยังใหม่อยู่หรือไม่ (ภายใน 1 วินาที)
        time_diff_dest = (current_time - self.last_destination_time).nanoseconds / 1e9
        time_diff_curr = (current_time - self.last_current_time).nanoseconds / 1e9
        
        if time_diff_dest > 1.0 or time_diff_curr > 1.0:
            self.get_logger().warn('Navigation data is stale, stopping vehicle')
            self.publish_stop_command()
            return
        
        # หา lookahead point
        lookahead_point = self.find_lookahead_point()
        
        # คำนวณการควบคุม (ใช้ angle-based control เป็นหลัก)
        steering_angle_rad = self.calculate_steering_angle(lookahead_point)
        throttle, brake, target_speed = self.calculate_speed_control(lookahead_point)
        
        # ส่งคำสั่งควบคุม
        self.publish_control_commands(steering_angle_rad, throttle, brake, target_speed)
        
        # Log information (ลดความถี่การ log)
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter % 20 == 0:  # log ทุก 1 วินาที (20 Hz / 20)
            line1_angle_deg = math.degrees(self.line1_angle)
            line2_angle_deg = math.degrees(self.line2_angle)
            angle_diff_deg = math.degrees(self.angle_difference)
            steering_angle_deg = math.degrees(steering_angle_rad)
            
            self.get_logger().info(
                f'Angles - Line1: {line1_angle_deg:.1f}°, Line2: {line2_angle_deg:.1f}°, '
                f'Diff: {angle_diff_deg:.1f}°, Steering: {steering_angle_deg:.1f}° ({steering_angle_rad:.3f} rad), '
                f'Speed: {target_speed:.1f} m/s (Fixed)'
            )
    
    def publish_control_commands(self, steering_angle_rad, throttle, brake, target_speed):
        """ส่งคำสั่งควบคุมไปยัง drive-by-wire system"""
        
        # Steering command - ส่งในหน่วยเรเดียน
        steering_msg = Float64()
        steering_msg.data = float(steering_angle_rad)  # เรเดียน
        self.steering_pub.publish(steering_msg)
        
        # Throttle command
        throttle_msg = Float64()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)
        
        # Brake command
        brake_msg = Float64()
        brake_msg.data = float(brake)
        self.brake_pub.publish(brake_msg)
        
        # Twist command (สำหรับ simulation)
        cmd_vel = Twist()
        cmd_vel.linear.x = float(target_speed)
        if target_speed > 0:
            cmd_vel.angular.z = float(steering_angle_rad * target_speed / self.wheelbase)
        else:
            cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def publish_stop_command(self):
        """ส่งคำสั่งหยุดรถ"""
        self.publish_control_commands(0.0, 0.0, 0.5, 0.0)

def main(args=None):
    rclpy.init(args=args)
    
    controller = PurePursuitController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Pure Pursuit Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()