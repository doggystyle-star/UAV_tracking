## This repository is based on Reinforcement Learning (DDPG), YOLOV5, Robust controller for UAV target tracking.

## Requirements
Python 3.10  
Tensorflow 2.14.0  
tensorlayer 2.2.5  

## File description
- gym_examples/: Custom Environment: The agent _(blue dot)_ navigates through both static  and dynamic obstacles _(black)_ to reach the goal _(green)_.
- ddpg_model/: The folder that stores the model weights in each episode.
- gym_examples/envs/gazebo_world.py: The detailed implementation of the env gazebo_world.
- gym_examples/envs/multirotor_communication.py: Main file1, Start the communication of UAV and the env gazebo_world.
- gym_examples/envs/tracking_IBVS.py: Main file2, the controller of the env gazebo_world.
- DDPG_UAV.py: Main file3， enhancing DDPG with Robust Controller for Accelerated Training.

## Results
"The DDPG algorithm is used for calcuating the parameter of robust controller, comeared to pure Reinforcement learning it is faster and using only cpu computing resources."

### **Please refer to the installation environment of this [PX4_YOLO](https://github.com/doggystyle-star/PX4_yolov5)**

## Start
Modify some launch files
* `cp -r gazeboworld/outdoor.launch* ~/PX4_Firmware/launch/ `
* `cp -r gazeboworld/world_nightmare.world* ~/PX4_Firmware/Tools/sitl_gazebo/worlds/ `

## Run
  * `roslaunch px4 outdoor.launch    # world`
  * `roslaunch yolov5_ros yolov5.launch    # find target`
  * `python3 multirotor_communication.py iris 0   # for communication`
  * `python3 multirotor_keboard_control.py iris 1 vel    # for control uav to move`
  
  * Close the file multirotor_keboard_control.py

  * `python3 tracking_IBVS.py # for tracking`  
  * `python3 DDPG_UAV.py    # for training`
  * `python3 DDPG_UAV.py --mode test --save_path /path/to/your/model      # for testing`

After about 700 episodes, the performance is shown as follows:

<div align="center">
  <img src="Robust_RL_test.gif" alt="result" width="50%" height="50%" />
</div>
