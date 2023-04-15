#!/bin/bash

# Get to main folder
cd ..

t=$1
echo $t

ROS_1_DISTRO=noetic
source "/opt/ros/$ROS_1_DISTRO/setup.bash"
. "../nav_noetic_ws/devel/setup.bash"
source "/opt/ros/foxy/setup.bash"
source install/setup.bash

# Open terminals or nohup
nohup=false

# Launch command
nbv_command="ros2 run publish_goal_cost nbv_training --ros-args -p time:=$t"
ent_command="ros2 run sogm_entropy sogm_entropy_node"

echo " "
echo "$nbv_command"
echo " "
echo "$ent_command"
echo " "

if [ "$nohup" = true ] ; then
    nohup $nbv_command
    nohup $ent_command
else
    xterm -bg black -fg lightgray -hold -T "NBV Node" -n "BV Node" -e $nbv_command &
    xterm -bg black -fg lightgray -hold -T "Entropy Node" -n "Entropy Node" -e $ent_command &

fi




















