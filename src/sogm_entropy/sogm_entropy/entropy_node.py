import sys

import rclpy
from rclpy.node import Node
from vox_msgs.msg import VoxGrid
import numpy as np



class SOGMEntropyNode(Node):

    def __init__(self):
        super().__init__('sogm_entropy')
        self.sogm_subscriber = self.create_subscription(
            VoxGrid,
            '/plan_costmap_3D',
            self.calc_entropy,
            10)


    def calc_entropy(self, msg: VoxGrid):
        sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0


        entropy = -sogm*np.log2(sogm) - (1-sogm)*np.log2(1-sogm)
        entropy[sogm == 0] = 0
        entropy[sogm == 1] = 0
        print(sogm.shape)
        print(np.mean(entropy, axis=(1, 2)))



def main(args=None):
    rclpy.init(args=args)

    minimal_client = SOGMEntropyNode()

    rclpy.spin(minimal_client)

    minimal_client.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()