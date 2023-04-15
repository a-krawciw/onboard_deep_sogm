import os
import random
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from publish_goal_cost.dqn import DQN

from publish_goal_cost.replay_memory import Transition, ReplayMemory

from .tf2_geo_msgs import do_transform_pose

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose, Point, PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import copy

import pickle as pk


from tf2_ros import TransformException, TransformStamped

from nav_msgs.msg import Odometry

from rclpy.duration import Duration
from rclpy.time import Time
import cv2
from cv_bridge import CvBridge



SOGM_DIM = 95
N_ACTIONS = 3
BATCH_SIZE = 10
GAMMA = 0.97716
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3
ENV_SIM = os.getenv('JACKAL_SIM_ROOT')


class NBVNode(Node):

    def __init__(self):
        super().__init__('sogm_nbv')
        print("NBV Node Started")


        self.time_folder = self.declare_parameter("time", "unknown")


        self.sogm_img_subscriber = self.create_subscription(
            Image,
            '/sogm_img',
            self.sogm_training,
            10)


        self.odom_sub = self.create_subscription(
            Odometry,
             '/ground_truth/state',
            self.store_robot_pose,
            10
        )

        self.nbv_via_point_pub = self.create_publisher(
            Path,
            '/move_base/TebLocalPlannerROS/via_points',
            10
        )

        self.base_meta_data = None
        self.steps_done = 0
        self.odom_loc = Pose()

        self.br = CvBridge()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(SOGM_DIM, N_ACTIONS)
        self.target_net = DQN(SOGM_DIM, N_ACTIONS)
        self.last_state = torch.zeros((1, SOGM_DIM, SOGM_DIM), dtype=torch.float32).to(self.device)
        self.total_reward = 0 #Save to a text file
        
        try:
            self.memory = pk.load(open(os.path.join(ENV_SIM, "onboard_deep_sogm", "rl_models", "memory_bank.pkl"), "rb"))
        except Exception as e:
            print(e)
            print("Pickle not loaded. Using empty memory")
            self.memory = ReplayMemory(500)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-3, amsgrad=True)

        try:
            with open(os.path.join(ENV_SIM, "onboard_deep_sogm", "rl_models", "1516_policy_net_final.pt"), 'rb') as model_file:
                model_params = torch.load(model_file)
                self.policy_net.load_state_dict(model_params)
                self.target_net.load_state_dict(model_params)
                print("Restored model weights")
        except FileNotFoundError as e:
            print(e)
            print("No model loaded. Using random weights")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    
    def select_action(self, state):
        self.steps_done += 1

        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            network_output = self.policy_net(state)
            action = network_output.max(1)[1].view(1, 1)

        return action

    def calc_reward(self, observation, action):
        eps = 1e-3 # Small value for numerical stability
        probability_field = (1. - 2 * eps) * observation + eps
        # Reward is sum of the negative entropy of all pixels
        reward = torch.sum(probability_field * torch.log(probability_field) +
                          (1. - probability_field) * torch.log(1. - probability_field))

        if action != 1:
            reward -= 100.0
        return torch.tensor([reward], device=self.device)

    def optimize_model(self):
        # Fill up our memory bank first before optimizing on batches
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def sogm_training(self, msg: Image):
       
        current_frame = self.br.imgmsg_to_cv2(msg).astype(np.float32)
        if (current_frame.size == 0):
            print("Warning converted image empty")
            return
        current_frame /= np.max(current_frame)
        observation = torch.tensor(current_frame, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        action = self.select_action(observation)
        reward = self.calc_reward(observation.squeeze(0), action)
        self.total_reward += reward.item()

        self.memory.push(self.last_state, action, observation, reward)

        self.last_state = observation

        self.optimize_model()

        self.target_net_state_dict = self.target_net.state_dict()
        self.policy_net_state_dict = self.policy_net.state_dict()
        for key in self.policy_net_state_dict:
            self.target_net_state_dict[key] = self.policy_net_state_dict[key]*TAU + self.target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(self.target_net_state_dict)

        print("Action:", action.item())
        self.publish_nbv_pose(action.item())

    def save_state(self):
        print(f"Total reward {self.total_reward}")
        #Save to txt
        file_path = ENV_SIM + "../Data/Simulation_v2/simulated_runs/" + self.time_folder + "/reward.txt"  # the file path and name

        # Open the file for writing
        with open(file_path, "w") as text_file:
           text_file.write(f"{self.total_reward}")

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
        nbv_pose.pose.position.y = 0.75 * (1 - action)

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

    minimal_client.save_state()
    minimal_client.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()