"""Minimal ROS 2 entry point to use rclpy and satisfy linters."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg  import Image
from cv_bridge import CvBridge

class VisualOdometry(Node):

    def __init__(self):
        super().__init__('visual_odom')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            msg_type=Image,
            topic= '/zed/zed_node/rgb_gray/image_rect_gray',
            callback=self.image_callback,
            qos_profile=10
        )

    def image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg)
        print(cv_image.shape)

def main(args=None) -> None:
    """Initialize and immediately shut down rclpy."""
    rclpy.init(args=args)
    vo_node = VisualOdometry()
    rclpy.spin(vo_node)
    vo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()