import sys

import rclpy
from rclpy.node import Node
from vox_msgs.msg import VoxGrid
from std_msgs.msg import Float64
import numpy as np


class SOGMEntropyNode(Node):
    def __init__(self):
        super().__init__('sogm_entropy')
        self.sogm_subscriber = self.create_subscription(
            VoxGrid,
            '/plan_costmap_3D',
            self.calc_entropy,
            10)
        
        self.sogm_publisher = self.create_publisher(
            Float64,
            '/sogm_entropy',
            10
        )
        print("SOGM Entropy Node Started")


    def calc_entropy(self, msg: VoxGrid):
        sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0
        entropy = -sogm*np.log2(sogm) - (1-sogm)*np.log2(1-sogm)
        entropy[sogm == 0] = 0
        entropy[sogm == 1] = 0
        print(sogm.shape)
        print(np.mean(entropy, axis=(1, 2)))

        sogm_entropy = np.sum(np.mean(entropy, axis=(1, 2)))
        print("SOGM Entropy: ", sogm_entropy)
        print("SOGM Entropy type: ", sogm_entropy.dtype)

        msg = Float64()
        msg.data = sogm_entropy

        self.sogm_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_client = SOGMEntropyNode()
    rclpy.spin(minimal_client)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()