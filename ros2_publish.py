# ros2_publish.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Point

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.pub_points = self.create_publisher(Point, 'navigation_points', 10)
        self.pub_roi_ids = self.create_publisher(Int32MultiArray, 'roi_mask_ids', 10)

    def publish_points(self, dest_point: Point, current_point: Point):
        self.pub_points.publish(dest_point)
        self.pub_points.publish(current_point)

    def publish_roi_ids(self, mask_ids):
        msg = Int32MultiArray()
        msg.data = mask_ids
        self.pub_roi_ids.publish(msg)
