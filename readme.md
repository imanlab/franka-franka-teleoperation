
Make sure you have franka_ros installed and working in your workspace

Add this package to your workspace

run the following to perform franka-franka teleoperation


```
roslaunch franka_hunter_gazebo pamda_follower.launch
```

```
rosrun framla_hunter_gazebo franka_two_arm_teleop
```

**Note:** Do not forget to make the franka_two_arm_teleop.py file executable by doing ```chmod +x franka_two_arm_teleop.py```


If you want to mimic grippers as well, run the following command

```
roslaunch franka_hunter_gazebo haptic_based_teleoperation.launch
```
