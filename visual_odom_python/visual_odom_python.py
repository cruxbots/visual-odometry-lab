"""Minimal ROS 2 entry point to use rclpy and satisfy linters."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # type: ignore
from nav_msgs.msg import Odometry # type: ignore
from nav_msgs.msg import Path as RosPath # type: ignore
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped #type: ignore
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from visual_odom_python.orb_vo import OrbVO

class VisualOdometry(Node):

    def __init__(self):
        super().__init__('visual_odom')

        self.bridge = CvBridge()
        self.image_sub_ = self.create_subscription(
            msg_type=Image,
            topic= '/camera/rgb/image_color',
            callback=self.image_callback,
            qos_profile=10
        )
        self.odom_pub_ = self.create_publisher(
            msg_type=Odometry,
            topic='vo_odom',
            qos_profile=10
        )
        self.path_pub_ = self.create_publisher(
            msg_type=RosPath,
            topic='vo_path',
            qos_profile=10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.orb_vo = OrbVO(
            intrinsic_param=Path("/home/rahul/vo_ws/src/visual_odom_python/visual_odom_python/config/rgbd_tum.txt")
            )
        self.rot_mat_global = np.eye(3)
        self.t_global = np.zeros((3,1))
        self.ros_path = RosPath()

    def image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        odom = Odometry()
        pose_stamped = PoseStamped()
        
        
        # Get rotation and translation between consecutive frames
        R, t = self.orb_vo.vo_runner(cv_image)
        if R is None:
            return
        # Get rot and translation from global frame to current
        R_prev = self.rot_mat_global
        self.rot_mat_global = R_prev @ R
        self.t_global = self.t_global + R_prev @ t
        # OpenCV uses optical frame (z-forward) and generally camera link frame (x-forward) is used (rep-105)
        # and hence have to transform from optical frame to camera link frame
        rot_mat_optical = Rot.from_matrix(self.rot_mat_global)
        rot_mat_optical_to_cam_link = Rot.from_euler('zx', [-90,90], degrees=True)
        rot_mat_cam_link = rot_mat_optical_to_cam_link * rot_mat_optical
        quat_cam_link = rot_mat_cam_link.as_quat()
        # TO DO:
        # Use from tf_transformations import quaternion_from_matrix might be better than scipy

        self.t_global_flat = self.t_global.flatten()
        t_cam_link = rot_mat_optical_to_cam_link.apply(self.t_global_flat)
        # inspired from https://stackoverflow.com/questions/74976911/create-an-odometry-publisher-node-in-python-ros2
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "vo_odom"
        odom.pose.pose.position.x = t_cam_link[0]
        odom.pose.pose.position.y = t_cam_link[1]
        odom.pose.pose.position.z = t_cam_link[2]
        odom.pose.pose.orientation.x = quat_cam_link[0]
        odom.pose.pose.orientation.y = quat_cam_link[1]
        odom.pose.pose.orientation.z = quat_cam_link[2]
        odom.pose.pose.orientation.w = quat_cam_link[3]
        odom.child_frame_id = "camera_link"
        self.odom_pub_.publish(odom)
        self.get_logger().info("published odom")

        # Gonna publish Path
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.ros_path.header = odom.header
        self.ros_path.poses.append(pose_stamped)
        self.path_pub_.publish(self.ros_path)

        # Gonna broadcast the TF
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "vo_odom"
        transform.child_frame_id = "camera_link"
        transform.transform.translation.x = t_cam_link[0]
        transform.transform.translation.y = t_cam_link[1]
        transform.transform.translation.z = t_cam_link[2]
        transform.transform.rotation.x = quat_cam_link[0]
        transform.transform.rotation.y = quat_cam_link[1]
        transform.transform.rotation.z = quat_cam_link[2]
        transform.transform.rotation.w = quat_cam_link[3]
        self.tf_broadcaster.sendTransform(transform)


def main(args=None) -> None:
    """Initialize and immediately shut down rclpy."""
    rclpy.init(args=args)
    vo_node = VisualOdometry()
    rclpy.spin(vo_node)
    vo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()