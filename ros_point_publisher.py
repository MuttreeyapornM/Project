#-----------------------------------for navigate.py [only first value] + pure_pursuit_controller.py------------
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PointStamped
# from std_msgs.msg import Header

# class PathPublisher(Node):
#     def __init__(self):
#         super().__init__('path_publisher')
#         self.pub_destination = self.create_publisher(PointStamped, 'destination_path', 10)
#         self.pub_current = self.create_publisher(PointStamped, 'current_path', 10)

#     def publish_points(self, destination_point, current_point):
#         now = self.get_clock().now().to_msg()

#         destination_msg = PointStamped(
#             header=Header(stamp=now, frame_id="map"),
#             point=destination_point
#         )
#         current_msg = PointStamped(
#             header=Header(stamp=now, frame_id="map"),
#             point=current_point
#         )

#         self.pub_destination.publish(destination_msg)
#         self.pub_current.publish(current_msg)



#---------------------for navigate_copy.py [multiple value] + pure_pursuit_controller_copy.py-------------------------
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PointStamped
# from std_msgs.msg import Header

# class PathPublisher(Node):
#     def __init__(self):
#         super().__init__('path_publisher')
#         self.pub_destination = self.create_publisher(PointStamped, 'line1_destination_path', 10)
#         self.pub_current = self.create_publisher(PointStamped, 'line2_current_path', 10)

#     def publish_points(self, destination_points, current_points):
#         now = self.get_clock().now().to_msg()

#         for dest_point in destination_points:
#             destination_msg = PointStamped(
#                 header=Header(stamp=now, frame_id="map"),
#                 point=dest_point
#             )
#             self.pub_destination.publish(destination_msg)

#         for curr_point in current_points:
#             current_msg = PointStamped(
#                 header=Header(stamp=now, frame_id="map"),
#                 point=curr_point
#             )
#             self.pub_current.publish(current_msg)


#-----------------------------------for navigate.py [only first value] + pure_pursuit_controller_filtered.py------------

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.pub_destination = self.create_publisher(Path, 'destination_path', 10)
        self.pub_current = self.create_publisher(Path, 'current_path', 10)

    def publish_points(self, destination_point, current_point):
        now = self.get_clock().now().to_msg()

        # ✅ กรณีที่รับแค่ 1 จุด Point → สร้าง Path ด้วย 1 จุด
        destination_path = Path()
        destination_path.header.stamp = now
        destination_path.header.frame_id = "map"

        dest_pose = PoseStamped()
        dest_pose.header.stamp = now
        dest_pose.header.frame_id = "map"
        dest_pose.pose.position = destination_point
        destination_path.poses.append(dest_pose)

        current_path = Path()
        current_path.header.stamp = now
        current_path.header.frame_id = "map"

        current_pose = PoseStamped()
        current_pose.header.stamp = now
        current_pose.header.frame_id = "map"
        current_pose.pose.position = current_point
        current_path.poses.append(current_pose)

        self.pub_destination.publish(destination_path)
        self.pub_current.publish(current_path)

