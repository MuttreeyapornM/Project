# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# import subprocess
# import time

# class SteeringNode(Node):
#     def __init__(self):
#         super().__init__('steering_node')
#         self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

#         self.last_sent_time = 0.0
#         self.message_interval = 0.5  # seconds
#         self.max_angular_speed = 1.5  # joystick max

#         # Send motor ON at startup
#         self.send_motor_on_message()

#     def send_motor_on_message(self):
#         toggle_message = "230D200100000000"  # Motor ON message
#         can_id = "06000001"
#         formatted_frame = f"{can_id}#{toggle_message}"
#         try:
#             subprocess.run(['cansend', 'can0', formatted_frame], check=True)
#             self.get_logger().info("Motor turned ON")
#         except Exception as e:
#             self.get_logger().error(f"Failed to turn on motor: {e}")

#     def cmd_vel_callback(self, msg):
#         current_time = time.time()
#         if current_time - self.last_sent_time < self.message_interval:
#             return

#         joystick_value = msg.angular.z / self.max_angular_speed
#         joystick_value = max(-1.0, min(1.0, joystick_value))  # Clamp

#         # scaled_value = int(joystick_value * 10000)
#         scaled_value = int(joystick_value * 10000 * 2.5)

#         scaled_value = max(-15000, min(15000, scaled_value))  # Limit range

#         if scaled_value < 0:
#             hex_value = hex((1 << 16) + scaled_value)[2:].upper()
#             direction = 'FFFF'
#         else:
#             hex_value = f"{scaled_value:04X}"
#             direction = '0000'

#         final_output = f"23022001{hex_value}{direction}"
#         can_id = "06000001"
#         formatted_frame = f"{can_id}#{final_output}"

#         try:
#             subprocess.run(['cansend', 'can0', formatted_frame], check=True)
#             self.get_logger().info(
#                 f"Sent CAN message: {formatted_frame} | angular.z: {msg.angular.z:.2f}"
#             )
#         except Exception as e:
#             self.get_logger().error(f"Failed to send CAN message: {e}")

#         self.last_sent_time = current_time

# def main(args=None):
#     rclpy.init(args=args)
#     node = SteeringNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()








# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# import subprocess
# import time
# import math  # ✅ ใช้สำหรับแปลง rad → deg

# class SteeringNode(Node):
#     def __init__(self):
#         super().__init__('steering_node')
#         self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

#         self.last_sent_time = 0.0
#         self.message_interval = 0.5  # ส่งทุก 0.5 วินาที
#         self.max_angle_deg = 50.0    # มุมสูงสุดที่พวงมาลัยทำได้ (องศา)

#         # ✅ ปรับ scale แยกฝั่งบวก/ลบ
#         self.scale_pos = 180   # ด้านขวา (บวก) - ตรงอยู่แล้ว
#         self.scale_neg = 320   # ด้านซ้าย (ลบ) - หมุนน้อย ต้องเพิ่ม scale เป็น 2 เท่า

#         # เปิดมอเตอร์ทันทีตอนเริ่มทำงาน
#         self.send_motor_on_message()

#     def send_motor_on_message(self):
#         toggle_message = "230D200100000000"  # Motor ON message
#         can_id = "06000001"
#         formatted_frame = f"{can_id}#{toggle_message}"
#         try:
#             subprocess.run(['cansend', 'can0', formatted_frame], check=True)
#             self.get_logger().info("✅ Motor turned ON")
#         except Exception as e:
#             self.get_logger().error(f"❌ Failed to turn on motor: {e}")

#     def cmd_vel_callback(self, msg):
#         current_time = time.time()
#         if current_time - self.last_sent_time < self.message_interval:
#             return

#         # ---------------- แปลงเรเดียน → องศา ----------------
#         angle_deg = math.degrees(msg.angular.z)

#         # จำกัดมุมให้อยู่ในช่วง -50° ถึง 50°
#         angle_deg = max(-self.max_angle_deg, min(self.max_angle_deg, angle_deg))

#         # ---------------- คำนวณค่า scaled ----------------
#         if angle_deg >= 0:
#             scaled_value = int(angle_deg * self.scale_pos)
#         else:
#             scaled_value = int(angle_deg * self.scale_neg)

#         # จำกัดค่าให้อยู่ในช่วง -15000 ถึง 15000
#         scaled_value = max(-15000, min(15000, scaled_value))

#         # ---------------- สร้างค่า hex สำหรับส่ง CAN ----------------
#         if scaled_value < 0:
#             hex_value = hex((1 << 16) + scaled_value)[2:].upper()
#             direction = 'FFFF'
#         else:
#             hex_value = f"{scaled_value:04X}"
#             direction = '0000'

#         final_output = f"23022001{hex_value}{direction}"
#         can_id = "06000001"
#         formatted_frame = f"{can_id}#{final_output}"

#         # ---------------- ส่งข้อมูลออกไปทาง CAN ----------------
#         try:
#             subprocess.run(['cansend', 'can0', formatted_frame], check=True)
#             self.get_logger().info(
#                 f"📤 Sent CAN message: {formatted_frame} | "
#                 f"angle_deg: {angle_deg:.2f}°, scaled: {scaled_value}"
#             )
#         except Exception as e:
#             self.get_logger().error(f"❌ Failed to send CAN message: {e}")

#         self.last_sent_time = current_time


# def main(args=None):
#     rclpy.init(args=args)
#     node = SteeringNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import subprocess
import time
import math
import os


class SteeringNode(Node):
    def __init__(self):
        super().__init__('steering_node')

        # 🔧 ตั้งค่า CAN0 bitrate = 250000
        self.setup_can_interface()

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            QoSProfile(depth=1)
        )
        self.feedback_subscription = self.create_subscription(
            Float64,
            '/steering/angle_feedback',
            self.steering_feedback_callback,
            10
        )

        self.last_sent_time = 0.0
        # Send steering commands fast enough for closed-loop path following.
        # 0.5s made the steering actuator lag several control cycles behind.
        self.message_interval = 0.05
        self.command_timeout = 0.3
        self.latest_cmd = Twist()
        self.latest_cmd_time = 0.0
        self.last_log_time = 0.0
        self.max_angle_deg = 50.0

        # scale ซ้ายขวา
        self.scale_pos = 180
        self.scale_neg = 320

        self.current_angle = 0.0
        self.feedback_angle_deg = None
        self.feedback_time = 0.0
        self.feedback_timeout = 0.5
        self.kp = 0.6
        self.ki = 0.0
        self.kd = 0.02
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_pid_time = None

        # เปิด motor ตอน start
        self.send_motor_on_message()
        self.command_timer = self.create_timer(
            self.message_interval, self.send_latest_command
        )

    # =====================================================
    # Setup CAN0
    # =====================================================
    def setup_can_interface(self):
        self.get_logger().info("🔧 Setting up CAN0 bitrate = 250000")

        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 250000')
        os.system('sudo ifconfig can0 txqueuelen 100000')
        os.system('sudo ifconfig can0 up')

        time.sleep(1.0)

    # =====================================================
    # Motor ON
    # =====================================================
    def send_motor_on_message(self):
        toggle_message = "230D200100000000"
        can_id = "06000001"
        formatted_frame = f"{can_id}#{toggle_message}"
        try:
            subprocess.run(['cansend', 'can0', formatted_frame], check=True)
            self.get_logger().info("✅ Motor turned ON")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to turn on motor: {e}")

    # =====================================================
    # Callback
    # =====================================================
    def steering_feedback_callback(self, msg):
        self.feedback_angle_deg = math.degrees(msg.data)
        self.feedback_time = time.time()

    def cmd_vel_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = time.time()

    def send_latest_command(self):
        current_time = time.time()

        if current_time - self.last_sent_time < self.message_interval:
            return

        # ==========================================
        # Target angle from cmd_vel
        # ==========================================
        if current_time - self.latest_cmd_time > self.command_timeout:
            target_angle = 0.0
        else:
            target_angle = math.degrees(self.latest_cmd.angular.z)
        target_angle = max(
            -self.max_angle_deg,
            min(self.max_angle_deg, target_angle)
        )

        # Use the navigation target directly. The actuator/controller should
        # handle the physical response; smoothing here adds avoidable lag.
        self.current_angle = target_angle
        angle_deg = self.compute_closed_loop_angle(target_angle, current_time)

        # ==========================================
        # scale
        # ==========================================
        if angle_deg >= 0:
            scaled_value = int(angle_deg * self.scale_pos)
        else:
            scaled_value = int(angle_deg * self.scale_neg)

        scaled_value = max(-15000, min(15000, scaled_value))

        # ==========================================
        # hex build
        # ==========================================
        if scaled_value < 0:
            hex_value = hex((1 << 16) + scaled_value)[2:].upper()
            direction = 'FFFF'
        else:
            hex_value = f"{scaled_value:04X}"
            direction = '0000'

        final_output = f"23022001{hex_value}{direction}"
        can_id = "06000001"
        formatted_frame = f"{can_id}#{final_output}"

        # ==========================================
        # send CAN
        # ==========================================
        try:
            subprocess.run(['cansend', 'can0', formatted_frame], check=True)
            if current_time - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f"📤 Sent CAN: {formatted_frame} | "
                    f"target: {target_angle:.2f}° | "
                    f"feedback: {self.format_feedback()} | "
                    f"cmd: {angle_deg:.2f}°"
                )
                self.last_log_time = current_time
        except Exception as e:
            self.get_logger().error(f"❌ Failed to send CAN message: {e}")

        self.last_sent_time = current_time

    def compute_closed_loop_angle(self, target_angle, current_time):
        if (
            self.feedback_angle_deg is None
            or current_time - self.feedback_time > self.feedback_timeout
        ):
            self.integral_error = 0.0
            self.last_pid_time = current_time
            return target_angle

        if self.last_pid_time is None:
            dt = self.message_interval
        else:
            dt = max(current_time - self.last_pid_time, self.message_interval)
        self.last_pid_time = current_time

        error = target_angle - self.feedback_angle_deg
        self.integral_error += error * dt
        self.integral_error = max(-20.0, min(20.0, self.integral_error))
        derivative = (error - self.last_error) / dt
        self.last_error = error

        correction = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * derivative
        )
        command_angle = target_angle + correction
        return max(-self.max_angle_deg, min(self.max_angle_deg, command_angle))

    def format_feedback(self):
        if self.feedback_angle_deg is None:
            return "None"
        if time.time() - self.feedback_time > self.feedback_timeout:
            return f"{self.feedback_angle_deg:.2f}° stale"
        return f"{self.feedback_angle_deg:.2f}°"


# =====================================================
# main
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = SteeringNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
