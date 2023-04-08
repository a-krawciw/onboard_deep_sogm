import sys

from .tf2_geo_msgs import do_transform_pose

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose, Point, PoseStamped
from vox_msgs.msg import VoxGrid
import numpy as np
import copy


from tf2_ros import TransformException, TransformStamped

from nav_msgs.msg import Odometry

from rclpy.duration import Duration
from rclpy.time import Time



class NBVNode(Node):

    def __init__(self):
        super().__init__('sogm_nbv')
        print("NBV Node Started")

        self.sogm_subscriber = self.create_subscription(
            VoxGrid,
            '/plan_costmap_3D',
            self.calc_entropy,
            10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.store_robot_pose,
            10
        )

        self.nbv_map_publisher = self.create_publisher(
            OccupancyGrid,
            '/plan_costmap_2D',
            10
        )

        self.nbv_via_point_pub = self.create_publisher(
            Path,
            '/move_base/TebLocalPlannerROS/via_points',
            10
        )

        self.base_meta_data = None
        self.odom_loc = Pose()


    def calc_entropy(self, msg: VoxGrid):
        self.sim_stamp = msg.header.stamp
        self.sogm = np.array(msg.data).reshape((msg.depth, msg.width, msg.height)) / 255.0
        if not self.base_meta_data:
            self.base_meta_data = MapMetaData()
            self.base_meta_data.height = msg.height
            self.base_meta_data.width = msg.width
            self.base_meta_data.resolution = msg.dl

        self.publish_cmap()

    def store_robot_pose(self, msg: Odometry):       
        self.odom_loc = msg.pose.pose
        self.T_r_m = TransformStamped()
        self.T_r_m.header = msg.header
        self.T_r_m.child_frame_id = 'base_link'
        self.T_r_m.transform.translation.x = msg.pose.pose.position.x
        self.T_r_m.transform.translation.y = msg.pose.pose.position.y
        self.T_r_m.transform.translation.z = msg.pose.pose.position.z
        self.T_r_m.transform.rotation = msg.pose.pose.orientation
        self.sim_stamp = msg.header.stamp

        self.publish_nbv_pose()
        
    def publish_nbv_pose(self):
        nbv_pose = PoseStamped()
        nbv_pose.header.frame_id = 'base_link'
        nbv_pose.header.stamp = self.sim_stamp
        nbv_pose.pose.position.x = 0.5
        nbv_pose.pose.position.y = 0.5

        print(self.sim_stamp)

        try:
            nbv_map = PoseStamped()
            nbv_map.header.frame_id = 'map'
            nbv_map.header.stamp = self.sim_stamp
            nbv_map.pose = do_transform_pose(nbv_pose.pose, self.T_r_m)
            nbv_wrapped_path = Path()
            nbv_wrapped_path.header = nbv_map.header
            nbv_wrapped_path.poses = [nbv_map]


            self.nbv_via_point_pub.publish(nbv_wrapped_path)
        except Exception as e:
            print("error")
            print(e)

        

    def publish_cmap(self):
        if not self.base_meta_data:
            return

        outgoing_msg = OccupancyGrid()
        outgoing_msg.header.frame_id = 'map'
        outgoing_msg.info = self.base_meta_data
        outgoing_msg.info.map_load_time = self.sim_stamp
        print(self.sim_stamp)

        outgoing_msg.info.origin.position.x = -8.0
        outgoing_msg.info.origin.position.y = -6.0
        cost_map = np.full((outgoing_msg.info.width, outgoing_msg.info.height), 0, dtype=int)
        cost_map[:outgoing_msg.info.width//4, outgoing_msg.info.height//4:3*outgoing_msg.info.height//4] = 40

        outgoing_msg.data = [int(a) for a in cost_map.ravel()]


        #self.nbv_map_publisher.publish(outgoing_msg)



def main(args=None):
    rclpy.init(args=args)

    
    minimal_client = NBVNode()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()