import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point
from vox_msgs.msg import VoxGrid
import numpy as np



class NBVNode(Node):

    def __init__(self):
        super().__init__('sogm_nbv')
        self.sogm_subscriber = self.create_subscription(
            VoxGrid,
            '/plan_costmap_3D',
            self.calc_entropy,
            10)

        self.nbv_map_publisher = self.create_publisher(
            OccupancyGrid,
            '/plan_costmap_2D',
            10
        )
        self.base_meta_data = None
        self.odom_loc = Pose()


    def calc_entropy(self, msg: VoxGrid):
        self.sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0
        self.store_robot_pose(msg)
        if not self.base_meta_data:
            self.base_meta_data = MapMetaData()
            self.base_meta_data.height = msg.height
            self.base_meta_data.width = msg.width
            self.base_meta_data.resolution = msg.dl

        self.publish_cmap()

    def store_robot_pose(self, msg: VoxGrid):
        self.odom_loc.position = msg.origin
        
        #In the SOGM z is time
        self.odom_loc.position.z = 0.0
        self.odom_loc.orientation.w = np.cos(msg.theta/2)
        self.odom_loc.orientation.z = np.sin(msg.theta/2)
        


    def publish_cmap(self):
        if not self.base_meta_data:
            return

        outgoing_msg = OccupancyGrid()
        outgoing_msg.info = self.base_meta_data
        outgoing_msg.info.map_load_time = self.get_clock().now().to_msg()
        outgoing_msg.info.origin = self.odom_loc
        cost_map = np.full((outgoing_msg.info.width, outgoing_msg.info.height), 60, dtype=int)
        cost_map[outgoing_msg.info.height//2:, outgoing_msg.info.height//4:3*outgoing_msg.info.height//4] = 0

        outgoing_msg.data = [int(a) for a in cost_map.ravel()]


        self.nbv_map_publisher.publish(outgoing_msg)



def main(args=None):
    rclpy.init(args=args)

    minimal_client = NBVNode()

    rclpy.spin(minimal_client)

    minimal_client.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()