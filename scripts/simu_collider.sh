#!/bin/bash

# Get to main folder
cd ..

ROS_1_DISTRO=noetic
source "/opt/ros/$ROS_1_DISTRO/setup.bash"
. "../nav_noetic_ws/devel/setup.bash"
source "/opt/ros/foxy/setup.bash"
# . "/opt/ros/foxy/setup.bash"
. install/setup.bash
# . "install/local_setup.bash"

# Check use_sim_time parameter
# rosparam set use_sim_time true


# Open terminals or nohup
nohup=true


# Old model path (dl=0.06 Sim only)
# trained_session="Log_2021-05-Bouncers"
# trained_session="Log_2021-05-Wanderers"
trained_session="Log_2022-06-22_16-29-03_A+H+Simu"
chkp_name="chkp_0240.tar"

# # Hybrid network (dl=0.12  ---  Training: real60% sim40%  ---  Time: 4s/40)
# log_name="Log_2022-03-01_16-47-49"
# chkp_name="chkp_0260.tar"

# Parameters
nav_without_sogm=false
use_sim_time=true
model_path="$PWD/../../SOGM-3D-2D-Net/results/$trained_session/checkpoints/$chkp_name"

# Launch command
sogm_command="ros2 launch deep_sogm sogm_launch.py"
sogm_command="$sogm_command nav_without_sogm:=$nav_without_sogm"
sogm_command="$sogm_command use_sim_time:=$use_sim_time"
sogm_command="$sogm_command model_path:=$model_path"

echo " "
echo "$sogm_command"
echo " "

if [ "$nohup" = true ] ; then

    # Start bridge in background and collider here
    nohup ros2 run ros1_bridge dynamic_bridge --bridge-all-2to1-topics > "nohup_bridge.txt" 2>&1 & $sogm_command

    # Start collider in background and bridge here 
    # nohup ros2 run deep_sogm collider > "nohup_sogm.txt" 2>&1 & ros2 run ros1_bridge dynamic_bridge

else

    # Start bridge in another terminal and collider here
    xterm -bg black -fg lightgray -e ros2 run ros1_bridge dynamic_bridge &
    $sogm_command

fi
































