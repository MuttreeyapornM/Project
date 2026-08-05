#------------------------steering angle in rad unit--------------------------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
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
        self.lookahead_distance_pixels = 100  # 100 pixels = 1.0 เมตร
        self.lookahead_distance = self.lookahead_distance_pixels * self.pixel_to_meter
        self.max_steering_angle_deg = 50.0  # องศา (สำหรับการตั้งค่า)
        self.max_steering_angle_rad = math.radians(self.max_steering_angle_deg)  # แปลงเป็นเรเดียน
        self.target_speed = 1.0  # m/s - FIXED SPEED
        self.control_frequency = 20.0  # Hz
        
        # Image coordinate system parameters
        self.image_width = 1280   # ความกว้างภาพ (ปรับตามภาพจริง)
        self.image_height = 720  # ความสูงภาพ (ปรับตามภาพจริง)
        self.image_center_x = self.image_width // 2
        self.image_center_y = self.image_height // 2
        
        # Subscribers - รับจาก navigate_ros.py
        self.destination_sub = self.create_subscription(
            Path, '/destination_path', self.destination_callback, 10)
        self.current_sub = self.create_subscription(
            Path, '/current_path', self.current_callback, 10)
        
        # Publishers สำหรับ drive-by-wire system
        self.steering_pub = self.create_publisher(Float64, '/vehicle/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, '/vehicle/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/vehicle/brake_cmd', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State variables
        self.smoothed_position = None
        self.alpha = 0.3  # EMA factor
        self.smoothed_heading = None
        self.last_angular_z = 0.0
        self.max_delta_angular = math.radians(5.0)  # 5 deg/frame
        self.destination_points = []  # Line1 points (destination path)
        self.current_points = []      # Line2 points (current position reference)
        self.current_position = None
        self.current_heading = 0.0
        self.last_destination_time = None
        self.last_current_time = None
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0/self.control_frequency, self.control_loop)
        
        self.get_logger().info('Pure Pursuit Controller initialized with FIXED SPEED: 1.0 m/s')
        self.get_logger().info(f'Max steering angle: {self.max_steering_angle_deg}° ({self.max_steering_angle_rad:.3f} rad)')
        self.get_logger().info('Waiting for destination_path (/destination_path) and current_path (/current_path) topics...')
        
    def destination_callback(self, msg):
        """รับจุดเป้าหมาย (Line1) จาก navigate_ros.py"""
        self.destination_points = []
        
        for pose in msg.poses:
            point_pixels = [pose.pose.position.x, pose.pose.position.y]
            point_meters = self.pixels_to_meters(point_pixels)
            self.destination_points.append(point_meters)
        
        self.last_destination_time = self.get_clock().now()
        
        # Debug log
        if len(self.destination_points) > 0:
            self.get_logger().info(f'Destination points (Line1): {len(self.destination_points)} points received')
    
    def current_callback(self, msg):
        """รับจุดอ้างอิง (Line2) จาก navigate_ros.py และใช้เป็นตำแหน่งปัจจุบัน"""
        self.current_points = []
        
        for pose in msg.poses:
            point_pixels = [pose.pose.position.x, pose.pose.position.y]
            point_meters = self.pixels_to_meters(point_pixels)
            self.current_points.append(point_meters)
        
        # ใช้จุดแรกเป็นตำแหน่งปัจจุบัน (หรือจุดที่ใกล้ที่สุดกับรถ)
        if self.current_points:
            current_raw = self.current_points[0]
            if self.smoothed_position is None:
                self.smoothed_position = current_raw
            else:
                self.smoothed_position = [
                    self.alpha * current_raw[0] + (1 - self.alpha) * self.smoothed_position[0],
                    self.alpha * current_raw[1] + (1 - self.alpha) * self.smoothed_position[1]
                ]
            self.current_position = self.smoothed_position
            
            # คำนวณ heading จาก 2 จุดแรกของ current_points
            if len(self.current_points) >= 2:
                dx = self.current_points[1][0] - self.current_points[0][0]
                dy = self.current_points[1][1] - self.current_points[0][1]
                heading_raw = math.atan2(dy, dx)
                if self.smoothed_heading is None:
                    self.smoothed_heading = heading_raw
                else:
                    delta = heading_raw - self.smoothed_heading
                    delta = math.atan2(math.sin(delta), math.cos(delta))
                    self.smoothed_heading += self.alpha * delta
                    self.smoothed_heading = math.atan2(math.sin(self.smoothed_heading), math.cos(self.smoothed_heading))
                self.current_heading = self.smoothed_heading
        
        self.last_current_time = self.get_clock().now()
        
        self.get_logger().debug(f'Current position (m): {self.current_position}')
    
    def pixels_to_meters(self, point_pixels):
        """แปลงจากพิกเซลเป็นเมตร"""
        # แปลงพิกเซลเป็นเมตร
        x_meters = point_pixels[0] * self.pixel_to_meter
        y_meters = point_pixels[1] * self.pixel_to_meter
        
        # แปลงพิกัด image coordinate (0,0 ที่มุมซ้ายบน) เป็น world coordinate
        # โดยให้จุดกึ่งกลางของภาพเป็น origin (0,0)
        x_world = x_meters - (self.image_center_x * self.pixel_to_meter)
        y_world = (self.image_center_y * self.pixel_to_meter) - y_meters  # flip Y axis
        
        return [x_world, y_world]
    
    def meters_to_pixels(self, point_meters):
        """แปลงจากเมตรเป็นพิกเซล (สำหรับ debug)"""
        x_pixels = point_meters[0] / self.pixel_to_meter
        y_pixels = point_meters[1] / self.pixel_to_meter
        return [x_pixels, y_pixels]
    
    def calculate_distance(self, point1, point2):
        """คำนวณระยะทางระหว่าง 2 จุด"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def find_lookahead_point(self):
        """หาจุด lookahead บน destination path (Line1)"""
        if not self.destination_points or not self.current_position:
            return None
            
        # หาจุดที่ใกล้ที่สุดบน path
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.destination_points):
            distance = self.calculate_distance(point, self.current_position)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # หาจุด lookahead โดยเริ่มจากจุดที่ใกล้ที่สุด
        for i in range(closest_index, len(self.destination_points)):
            point = self.destination_points[i]
            distance = self.calculate_distance(point, self.current_position)
            
            if distance >= self.lookahead_distance:
                return point
        
        # ถ้าไม่เจอ ใช้จุดสุดท้าย
        if self.destination_points:
            return self.destination_points[-1]
        
        return None
    
    def calculate_steering_angle(self, lookahead_point):
        """
        คำนวณมุมพวงมาลัยด้วย Pure Pursuit Algorithm
        ผลลัพธ์ในหน่วยเรเดียน โดย:
        - 0 = ตรง
        - บวก = เลี้ยวซ้าย
        - ลบ = เลี้ยวขวา
        """
        if not self.current_position or not lookahead_point:
            return 0.0
        
        # คำนวณ vector ไปยัง lookahead point
        dx = lookahead_point[0] - self.current_position[0]
        dy = lookahead_point[1] - self.current_position[1]
        
        # คำนวณ angle ไปยัง lookahead point
        target_angle = math.atan2(dy, dx)
        
        # คำนวณ lateral error (alpha) - มุมที่ต้องเลี้ยว
        alpha = target_angle - self.current_heading
        
        # Normalize angle to [-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        
        # คำนวณ lookahead distance
        ld = math.sqrt(dx**2 + dy**2)
        
        if ld < 0.1:  # หลีกเลี่ยง division by zero
            return 0.0
        
        # Pure Pursuit formula - ผลลัพธ์อยู่ในหน่วยเรเดียน
        steering_angle_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        
        # จำกัดมุมพวงมาลัย (ในหน่วยเรเดียน)
        steering_angle_rad = max(-self.max_steering_angle_rad, 
                                min(self.max_steering_angle_rad, steering_angle_rad))
        
        return steering_angle_rad
    
    def calculate_speed_control(self, lookahead_point, steering_angle_rad):
        """ควบคุมความเร็วแบบคงที่ 1.0 m/s"""
        if lookahead_point is None:
            # ไม่มีจุดถัดไป -> หยุดรถ
            throttle = 0.0
            brake = 1.0
            target_speed = 0.0
        else:
            # ใช้ความเร็วคงที่ 1.0 m/s
            target_speed = self.target_speed  # 1.0 m/s
            
            # Simple throttle control for constant speed
            throttle = 0.5  # Fixed throttle for 1.0 m/s
            brake = 0.0

        return throttle, brake, target_speed
    
    def control_loop(self):
        """หลัก control loop"""
        # ตรวจสอบว่ามีข้อมูลล่าสุดหรือไม่
        current_time = self.get_clock().now()
        
        if (self.last_destination_time is None or 
            self.last_current_time is None):
            self.get_logger().warn('Waiting for navigation data...')
            return
        
        # ตรวจสอบว่าข้อมูลยังใหม่อยู่หรือไม่ (ภายใน 2 วินาที)
        time_diff_dest = (current_time - self.last_destination_time).nanoseconds / 1e9
        time_diff_curr = (current_time - self.last_current_time).nanoseconds / 1e9
        
        if time_diff_dest > 2.0 or time_diff_curr > 2.0:
            self.get_logger().warn('Navigation data is stale, stopping vehicle')
            self.publish_stop_command()
            return
        
        # หา lookahead point
        lookahead_point = self.find_lookahead_point()
        if not lookahead_point:
            self.get_logger().warn('No lookahead point found')
            self.publish_stop_command()
            return
        
        # คำนวณการควบคุม
        steering_angle_rad = self.calculate_steering_angle(lookahead_point)
        throttle, brake, target_speed = self.calculate_speed_control(lookahead_point, steering_angle_rad)
        
        # ส่งคำสั่งควบคุม
        self.publish_control_commands(steering_angle_rad, throttle, brake, target_speed)
        
        # Log information (ลดความถี่การ log)
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter % 40 == 0:  # log ทุก 2 วินาที (20 Hz / 40)
            distance_m = self.calculate_distance(lookahead_point, self.current_position)
            distance_pixels = distance_m / self.pixel_to_meter
            steering_angle_deg = math.degrees(steering_angle_rad)
            
            # Determine steering direction
            if abs(steering_angle_deg) < 1:
                direction = "STRAIGHT"
            elif steering_angle_deg > 0:
                direction = "LEFT"
            else:
                direction = "RIGHT"
            
            self.get_logger().info(
                f'Control - Steering: {steering_angle_deg:.1f}° ({steering_angle_rad:.3f} rad) [{direction}], '
                f'Speed: {target_speed:.1f} m/s [FIXED], Throttle: {throttle:.2f}, '
                f'Brake: {brake:.2f}, Distance: {distance_m:.2f}m ({distance_pixels:.0f}px)'
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
        
        # Twist command (สำหรับ low-level control)
        cmd_vel = Twist()
        cmd_vel.linear.x = float(target_speed)  # ความเร็วเชิงเส้น (m/s) - FIXED 1.0 m/s
        
        # Angular velocity for steering (rad/s)
        # ใช้ bicycle model: ω = v * tan(δ) / L
        # โดย δ = steering angle, L = wheelbase, v = speed
        if target_speed > 0.1 and abs(steering_angle_rad) > 0.001:
            # ใช้ simple kinematic model
            omega = float(target_speed * math.tan(steering_angle_rad) / self.wheelbase)
            max_angular_vel = math.radians(60)
            omega = max(-max_angular_vel, min(max_angular_vel, omega))
            delta = omega - self.last_angular_z
            delta = max(-self.max_delta_angular, min(self.max_delta_angular, delta))
            self.last_angular_z += delta
            cmd_vel.angular.z = self.last_angular_z
            
            # จำกัด angular velocity
            max_angular_vel = math.radians(60)  # 60 deg/s
            cmd_vel.angular.z = max(-max_angular_vel, min(max_angular_vel, cmd_vel.angular.z))
        else:
            cmd_vel.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def publish_stop_command(self):
        """ส่งคำสั่งหยุดรถ"""
        self.publish_control_commands(0.0, 0.0, 1.0, 0.0)

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