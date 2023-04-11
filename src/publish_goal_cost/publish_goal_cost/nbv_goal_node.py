import os
import sys

import torch
from publish_goal_cost.dqn import DQN

from .tf2_geo_msgs import do_transform_pose

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose, Point, PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import copy


from tf2_ros import TransformException, TransformStamped

from nav_msgs.msg import Odometry

from rclpy.duration import Duration
from rclpy.time import Time
import cv2
from cv_bridge import CvBridge



SOGM_DIM = 95
N_ACTIONS = 3
ENV_SIM = os.getenv('JACKAL_SIM_ROOT')


class NBVNode(Node):

    def __init__(self):
        super().__init__('sogm_nbv')
        print("NBV Node Started")

        self.sogm_img_subscriber = self.create_subscription(
            Image,
            '/sogm_img',
            self.sogm_inference,
            10)

        # self.sogm_subscriber = self.create_subscription(
        #     VoxGrid,
        #     '/plan_costmap_3D',
        #     self.calc_entropy,
        #     10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.store_robot_pose,
            10
        )

        # self.nbv_map_publisher = self.create_publisher(
        #     OccupancyGrid,
        #     '/plan_costmap_2D',
        #     10
        # )

        self.nbv_via_point_pub = self.create_publisher(
            Path,
            '/move_base/TebLocalPlannerROS/via_points',
            10
        )

        self.base_meta_data = None
        self.odom_loc = Pose()

        self.br = CvBridge()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(SOGM_DIM, N_ACTIONS)

        try:
            with open(os.path.join(ENV_SIM, "onboard_deep_sogm", "src", "rl_models", "1516_policy_net.pt")) as model_file:
                model_params = torch.load(model_file)
                self.policy_net.load_state_dict(model_params)
        except FileNotFoundError as e:
            print(e)
            print("No model loaded. Using random weights")
        self.policy_net.to(self.device)


    def sogm_inference(self, msg: Image):
       
        current_frame = self.br.imgmsg_to_cv2(msg).astype(np.float32)
        if (current_frame.size == 0):
            print("Warning converted image empty")
            return
        current_frame /= np.max(current_frame)
        observation = torch.tensor(current_frame, dtype=torch.float32)
        with torch.no_grad():
            network_output = self.policy_net(observation.unsqueeze(0).to(self.device))
            action = network_output.max(1)[1].view(1, 1).item()
            print("Action:", action)

            self.publish_nbv_pose(action)


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

        
    def publish_nbv_pose(self, action: int):
        nbv_pose = PoseStamped()
        nbv_pose.header.frame_id = 'base_link'
        nbv_pose.header.stamp = self.sim_stamp
        nbv_pose.pose.position.x = 0.5
        nbv_pose.pose.position.y = 0.5 * (action - 1)

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



def main(args=None):
    rclpy.init(args=args)

    
    minimal_client = NBVNode()
    rclpy.spin(minimal_client)

    minimal_client.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()