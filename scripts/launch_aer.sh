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


# Parameters

# Launch command
nbv_command="ros2 run publish_goal_cost nbv_goal_node"

echo " "
echo "$nbv_command"
echo " "

if [ "$nohup" = true ] ; then

    # Start bridge in background and collider here
    nohup $nbv_command

    # Start collider in background and bridge here 
    # nohup ros2 run deep_sogm collider > "nohup_sogm.txt" 2>&1 & ros2 run ros1_bridge dynamic_bridge

else

    # Start bridge in another terminal and collider here
    xterm -bg black -fg lightgray -e $nbv_command

fi




















