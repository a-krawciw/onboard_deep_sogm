import sys

import rclpy
from rclpy.node import Node
from vox_msgs.msg import VoxGrid
from std_msgs.msg import Float64, Header
from sensor_msgs.msg import Image
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

        self.sogm_img_publisher = self.create_publisher(
            Image,
            '/sogm_img',
            10
        )
        print("SOGM Entropy Node Started")


    def calc_entropy(self, msg: VoxGrid):
        sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0
        entropy = -sogm*np.log2(sogm) - (1-sogm)*np.log2(1-sogm)
        entropy[sogm == 0] = 0
        entropy[sogm == 1] = 0
        print(sogm.shape)
        print(msg.dt)

        sogm_entropy = np.sum(np.mean(entropy, axis=(1, 2)))
        print("SOGM Entropy: ", sogm_entropy)
        print("SOGM Entropy type: ", sogm_entropy.dtype)
        #print(np.amax(msg.data))
        #print(np.amin(msg.data))

        # Collapse the third axis (time) of the data
        sogm_3d = np.array(msg.data).reshape((msg.depth, msg.width, msg.height))
        sogm_2d = np.sum(sogm_3d, axis=0)
        #sogm_2d = sogm_3d[0, :, :]
        # Find max and min values in the array
        max_val = np.amax(sogm_2d)
        min_val = np.amin(sogm_2d)
        print("Max: ", max_val)
        print("Min: ", min_val)
        # Normalize the array to be between 0 and 255
        #sogm_2d_pub = (sogm_2d - min_val) / (max_val - min_val) * 255
        # Clip to 255
        sogm_2d_pub = np.clip(sogm_2d, 0, 255)
        # Convert to uint8
        sogm2d_pub = sogm_2d_pub.astype(np.uint8)
        print(sogm_2d.shape)

        # For debugging, let's generate a 20x20 array with the inner 5x5 cells being 255 and all others being 0
        #sogm2d_pub = np.zeros((20, 20), dtype=np.uint8)
        #sogm2d_pub[7:12, 7:12] = 255

        # For visualization purposes, add a 255 valued border around the array
        sogm2d_pub[0, :] = 255
        sogm2d_pub[-1, :] = 255
        sogm2d_pub[:, 0] = 255
        sogm2d_pub[:, -1] = 255

        # Create the Image message
        img_msg = Image()
        img_msg.header = Header()
        img_msg.height = sogm2d_pub.shape[0]
        img_msg.width = sogm2d_pub.shape[1]
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = 0
        #img_msg.step = sogm2d_pub.shape[1] # 1 byte per pixel
        img_msg.data = sogm2d_pub.tostring()



        pub_msg = Float64()
        pub_msg.data = sogm_entropy

        self.sogm_publisher.publish(pub_msg)
        self.sogm_img_publisher.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_client = SOGMEntropyNode()
    rclpy.spin(minimal_client)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()