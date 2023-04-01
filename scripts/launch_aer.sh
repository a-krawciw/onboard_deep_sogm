#!/bin/bash

# Get to main folder
cd ..

ROS_1_DISTRO=noetic
source "/opt/ros/$ROS_1_DISTRO/setup.bash"
. "../nav_noetic_ws/devel/setup.bash"
source "/opt/ros/foxy/setup.bash"
source install/setup.bash

# Open terminals or nohup
nohup=true

# Launch command
nbv_command="ros2 run publish_goal_cost nbv_goal_node"
ent_command="ros2 run sogm_entropy sogm_entropy_node"

echo " "
echo "$nbv_command"
echo " "
echo "$ent_command"
echo " "

if [ "$nohup" = true ] ; then
    #nohup $nbv_command
    nohup $ent_command
else
    xterm -bg black -fg lightgray -e $nbv_command
    xterm -bg black -fg lightgray -e $ent_command

fi




















