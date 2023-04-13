import sys

import rclpy
from rclpy.node import Node
from vox_msgs.msg import VoxGrid
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R


class SOGMEntropyNode(Node):
    def __init__(self):
        super().__init__('sogm_entropy')
        # Subscribers
        self.sogm_subscriber = self.create_subscription(
            VoxGrid,
            '/plan_costmap_3D',
            self.process_SOGM,
            10)
        
        self.state_subscriber = self.create_subscription(
            Odometry,
            '/ground_truth/state',
            self.update_pose,
            10)

        # Publishers
        self.sogm_publisher = self.create_publisher(
            Float32,
            '/sogm_entropy',
            10
        )

        self.sogm_img_publisher = self.create_publisher(
            Image,
            '/sogm_img',
            10
        )

        # Initialize pose
        self.pose = None
        self.yaw = 0.0      # Yaw of robot in degrees
        print("SOGM Entropy Node Started")

    def update_pose(self, msg):
        self.pose = msg.pose.pose
        # Convert quaternion to euler angles
        q = self.pose.orientation
        euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True)
        self.yaw = euler[2]
        #self.yaw = 0.0

    def process_SOGM(self, msg: VoxGrid):
        print("Received SOGM message")

        # Extract raw 3D SOGM
        sogm_3d = np.array(msg.data).reshape((msg.depth, msg.width, msg.height))

        # Extract the 2D SOGM
        sogm_2d = self.process_3D_SOGM(sogm_3d, msg.dt)

        # Publish SOGM entropy
        self.pub_entropy(sogm_2d)

        # Publish SOGM image
        self.pub_img(sogm_2d)

    def process_3D_SOGM(self, sogm_3d, dt):
        sogm_3d_mod = sogm_3d.copy()
        # The first layer corresponds to static obstacles, so we can ignore it
        sogm_3d_mod = sogm_3d_mod[1:, :, :]

        # Downweight each layer by dt
        time_factor = 0.9
        min_risk_val = 180
        for ii in range(sogm_3d_mod.shape[0]):
            # Set all values below the minimum risk value to 0
            sogm_2d_ii = sogm_3d_mod[ii, :, :]
            sogm_2d_ii[sogm_2d_ii < min_risk_val] = 0
            if ii > 1:
                sogm_3d_mod[ii, :, :] = sogm_2d_ii * (time_factor ** ii)

        # Find the maximum value along the time axis
        sogm_2d = np.max(sogm_3d_mod, axis=0)

        # Flip about x axis to align with world frame y axis
        # This is because 0,0 in matrix is top left whereas 0,0 in world frame is bottom left
        sogm_2d = np.flip(sogm_2d, axis=0)

        # Rotate the array to align with robot frame
        sogm_2d = ndimage.rotate(sogm_2d, -self.yaw, reshape=False)

        # Clip to 255 to be safe
        sogm_2d = np.clip(sogm_2d, 0, 255)

        return sogm_2d

    def pub_entropy(self, sogm_2d):
        #sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0
        #entropy = -sogm*np.log2(sogm) - (1-sogm)*np.log2(1-sogm)
        #entropy[sogm == 0] = 0
        #entropy[sogm == 1] = 0
        #sogm_entropy = np.sum(np.mean(entropy, axis=(1, 2)))

        # Convert to probabilities
        sogm_2d = sogm_2d / 255.0 + 1e-6

        # Compute the marginal distributions
        row_probs = np.sum(sogm_2d, axis=1)
        col_probs = np.sum(sogm_2d, axis=0)
        # Compute the row and column entropies separately
        row_entropy = -np.sum(row_probs * np.log2(row_probs))
        col_entropy = -np.sum(col_probs * np.log2(col_probs))
        # Compute the joint entropy by reshaping the matrix into a 1D array
        joint_entropy = -np.sum((sogm_2d * np.log2(sogm_2d)).flatten())

        print(joint_entropy)
        # Publish the entropy
        pub_msg = Float32()
        pub_msg.data = joint_entropy
        self.sogm_publisher.publish(pub_msg)

    def pub_img(self, sogm_2d):
        # For debugging, let's generate a 20x20 array with the inner 5x5 cells being 255 and all others being 0
        #sogm2d_pub = np.zeros((20, 20), dtype=np.uint8)
        #sogm2d_pub[7:12, 7:12] = 255

        # For debugging, let's generate a 20x20 array with the top right 5x5 cells being 255 and all others being 0
        #sogm_2d = np.zeros((20, 20), dtype=np.uint8)
        #sogm_2d[0:5, 15:20] = 255
        #sogm_2d = ndimage.rotate(sogm_2d, 180.0, reshape=False)

        # Convert to uint8
        sogm2d_pub = sogm_2d.astype(np.uint8)

        # The current sogm_2d is resolved in robot frame, with y up and x to the right
        # For visualization purposes, let's point x upwards
        sogm2d_pub = ndimage.rotate(sogm2d_pub, 90.0, reshape=False)

        # Add a 255 valued border around the array
        # sogm2d_pub[0, :] = 255
        # sogm2d_pub[-1, :] = 255
        # sogm2d_pub[:, 0] = 255
        # sogm2d_pub[:, -1] = 255

        # Create the Image message
        img_msg = Image()
        img_msg.header = Header()
        img_msg.height = sogm2d_pub.shape[0]
        img_msg.width = sogm2d_pub.shape[1]
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = 0
        img_msg.step = sogm2d_pub.shape[1] # 1 byte per pixel
        img_msg.data = sogm2d_pub.tostring()

        # Publish the image
        self.sogm_img_publisher.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    minimal_client = SOGMEntropyNode()
    rclpy.spin(minimal_client)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()