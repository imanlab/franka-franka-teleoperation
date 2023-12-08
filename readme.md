Make sure two PCs are able to communicate with each other using ping



If so and if you are on the leader PC add this to your .bashrc

```
export ROS_MASTER_URI=http://{LEADER_IP}:11311
export ROS_IP={LEADER_IP}
```

If so, add the following lines at the end of your .bashrc file

```
export ROS_MASTER_URI=http://{LEADER_IP}:11311
export ROS_IP={FOLLOWER_IP}
```

Replace LEADER_IP and FOLLOWER_IP above with the static ip addresses om your systems.
Now you can communicate between two PCs using ROS.

**Note:** Make sure you launch the files in the leader workspace before running the follower commands

Look at the **leader_franka** and **follower_franka** branches within this repository for more information

