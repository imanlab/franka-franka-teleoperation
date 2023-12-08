#!/usr/bin/env python3

import numpy as np
import os 
import matplotlib.pyplot as plt
import time

from tf.transformations import *

# quat = quaternion_from_euler(90,0,0)
# quat = euler_from_quaternion([0.85090352, 0, 0, 0.52532199])
# print("Quaternion: ",quat)
# x =  np.arange(0, 2*np.pi, (2*np.pi)/100)
# y = np.sin(x)
# gradient = np.gradient(y)/np.gradient(x)
# tangent = np.tan(y)

# Take previous and next point and draw a line. This will be the tangemt. Take angle to that line 


# pt_1 = -5
# pt_2 = 5
# tangent = gradient*(x-3) + 0.5
# tangent_x = np.linspace(pt_1-1, pt_2+1, 100)
# plt.plot(x, y, 'b',x, tangent, 'r--')
# plt.show()


# Moving average of haptic feedback

def moving_average(arrayName):
	window_size = 10
	i = 1
	moving_average_feedback = []
	for j in range(window_size):
		moving_average_feedback.append(arrayName[j])

	while i < len(arrayName) - window_size + 1:
		window = arrayName[ i: i+window_size ]
		window_average = round(sum(window)/ window_size, 3)

		moving_average_feedback.append(window_average)
		i += 1
	return moving_average_feedback

def joints_array(joint_name, joint_number):
	val=0
	joints_array = []
	while val < len(joint_name):
		joints_array.append(joint_name[val][joint_number-1])
		val += 1
	return joints_array



# Plot x direction and haptic feedback
def no_subplot_fn():
	plt.figure(1, figsize=(12,3))
	plt.subplot(411)
	plt.plot(time, ref_x - actual_x, 'b--', label = 'Difference along x axis', linewidth=2.0)
	plt.subplot(412)
	plt.plot(time, ref_y - actual_y, 'b--', label = 'Difference along y axis', linewidth=2.0)
	plt.subplot(413)
	plt.plot(time, moving_average_haptic_x, 'r', label = 'Haptic feedback', linewidth=2.0)
	plt.subplot(414)
	plt.plot(time, moving_average_haptic_yaw, 'r', label = 'Haptic feedback yaw', linewidth=2.0)
	plt.show()


# Plot all 4 using plt.subplots() function
def with_subplot_fn():
	fig, axis = plt.subplots(2,2, figsize = (12,3), sharey = True)
	axis[0,0].plot(time, ref_x - actual_x, linewidth=2.0)
	axis[0,0].set_title("Difference along x axis")

	axis[1,0].plot(time, moving_average_haptic_x, linewidth=2.0)
	axis[1,0].set_title("Linear haptic feedback")

	axis[0,1].plot(time, ref_y - actual_y, linewidth=2.0)
	axis[0,1].set_title("Difference along y axis")

	axis[1,1].plot(time, moving_average_haptic_yaw, linewidth=2.0)
	axis[1,1].set_title("Angular haptic feedback")

	# for axis in axis.flat:
	# 	axis.set(xlabel = 'Time (sec)', ylabel='Difference in Trajectories (m)')

	plt.setp(axis[-1,:], xlabel='Time(sec)')
	plt.setp(axis[0,:], ylabel='Difference in Trajectories (m)')
	plt.setp(axis[1,:], ylabel='Haptic feedback (N)')
	plt.show()

def with_subplot_obstacle():
	fig, axis = plt.subplots(2,3, figsize = (3,8))

	axis[0,0].plot(time, ref_x - actual_x, 'y',linewidth=2.0)
	axis[0,0].set_title("Difference along x axis")
	axis[1,0].plot(time, moving_average_haptic_x, 'g', linewidth=2.0)
	axis[1,0].set_title("Linear haptic feedback")

	axis[0,1].plot(time, ref_x - actual_x, 'y',linewidth=2.0)
	axis[0,1].set_title("Difference along x axis")
	axis[1,1].plot(time_obstacle, moving_average_obstacle, 'r',linewidth=2.0)
	axis[1,1].set_title("Obstacle haptic feedback")

	axis[0,2].plot(time, ref_y - actual_y, 'y',linewidth=2.0)
	axis[0,2].set_title("Difference along y axis")
	axis[1,2].plot(time, moving_average_haptic_yaw, 'b',linewidth=2.0)
	axis[1,2].set_title("Angular haptic feedback")

	

	plt.setp(axis[-1,:], xlabel='Time(sec)')
	plt.setp(axis[0,:], ylabel='Difference in Trajectories (m)')
	plt.setp(axis[1,:], ylabel='Haptic feedback (N)')

	plt.show()
# plt.plot(time, ref_x, 'r--', label = 'Reference Trajectory', linewidth=2.0)
# plt.plot(time, actual_x, 'g--', label = "Actual trajectory followed", linewidth=2.0)
# plt.plot(time, moving_average_haptic_x, 'b', label = 'Haptic feedback', linewidth=2.0)
# plt.plot(time, moving_average_haptic_yaw, 'yo', label = 'Haptic feedback yaw', linewidth=2.0)
# plt.plot(time, ref_x - actual_x, 'b--', label = 'Difference in trajectory', linewidth=2.0)
# plt.plot(goal[0], marker= 'o', markerfacecolor='green')

# plt.ylim(-7, 7)
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.xlabel('Time (sec)', fontsize = 16)
# plt.ylabel('Trajectories (m)', fontsize=16)
# plt.legend(loc='upper left', fontsize = 14)
# plt.title('Time vs Trajectories along x direction', fontsize = 16)

# plt.show()

if __name__ == '__main__':
    file = np.load('/home/venkatesh/franka_hunter_ws/src/MobileManip_franka_hunter/franka_hunter_gazebo/script/real_h_3.npz')

    haptic = True
    distractor = False
    if haptic == True:
        ref_x = file['reference_x']
        actual_x = file['actual_x']
        ref_y = file['reference_y']
        actual_y = file['actual_y']
        feedback = file['haptic_feedback_x']
        external_torque = file['external_torques']
        leader_joints = file['leader_joint_states']
        follower_joints = file['follower_joint_states']
        time = file['mobile_time']
        follower_time = file['follower_time']

        steps = np.arange(len(ref_x))
        time = np.linspace(0, file['mobile_time'][1] - file['mobile_time'][0], num=len(file['reference_x']))

        obstacle_feedback = file['obstacle_haptic_force']
        time_obstacle = np.linspace(0, file['mobile_time'][1] - file['mobile_time'][0], num=len(file['obstacle_haptic_force']))

        time_torque = np.linspace(0, file['mobile_time'][1] - file['mobile_time'][0], num=len(file['external_torques']))

        time_leader_joints = np.linspace(0, file['mobile_time'][1] - file['mobile_time'][0], num=len(file['leader_joint_states']))
        time_follower_joints = np.linspace(0, file['mobile_time'][1] - file['mobile_time'][0], num=len(file['follower_joint_states']))

        moving_average_haptic_x = moving_average(file['haptic_feedback_x'])
        moving_average_haptic_yaw = moving_average(file['haptic_feedback_y'])
        moving_average_obstacle = moving_average(obstacle_feedback)
        moving_average_ext_torques = moving_average(external_torque)

        leader_joints = joints_array(leader_joints, 7)
        moving_average_leader_joints = moving_average(leader_joints)
        follower_joints = joints_array(follower_joints, 7)
        moving_average_follower_joints = moving_average(follower_joints)
        # goal = file['goal']

        print("ACTUAL_y", len(file['actual_y']))
        print("ACTUAL_x", len(file['actual_x']))
        print("ACTUAL_z", len(file['actual_z']))
        print("REFERENCE_y", len(file['reference_y']))
        print("REFERENCE_x", len(file['reference_x']))
        print("REFERENCE_z", len(file['reference_z']))
        print("HAPTIC_X", len(file['haptic_feedback_x']))
        print("HAPTIC_Y", len(file['haptic_feedback_y']))
        print("HAPTIC_Z", len(file['haptic_feedback_z']))
        print("FOLLOWER EXTERNAL TORQUES", len(file['follower_external_torque']))
        print("FOLLOWER TIME", len(file['follower_time']))
        print("FOLLOWER GOAL", len(file['follower_goal']))

        print("########### Mobile Robot ##################")
        print("MOBILE_REFERENCE_x", len(file['mobile_reference_x']))
        print("MOBILE_REFERENCE_y", len(file['mobile_reference_y']))
        print("MOBILE_ACTUAL_x", len(file['mobile_actual_x']))
        print("MOBILE_ACTUAL_y", len(file['mobile_actual_y']))
        print("MOBILE_HAPTIC_X", len(file['mobile_haptic_feedback_x']))
        print("MOBILE_HAPTIC_Y", len(file['mobile_haptic_feedback_yaw']))
        print("OBSTACLE_HAPTIC", len(file['obstacle_haptic_force']))
        print("EXTERNAL TORQUES", len(file['external_torques']))
        print("MOBILE TIME", len(file['mobile_time']))
        print("MOBILE GOAL", len(file['mobile_goal']))

        print("")
        # print("LEADER JOINTS", len(file['leader_joint_states']))
        # print("FOLLOWER JOINTS", len(file['follower_joint_states']))
        # print("DISTRACTORS", len(file['distractors']))
        # print(file['distractors'])

        print("NET Time: ", file['mobile_time'][1] - file['mobile_time'][0])
        print("NET Follower time: ",file['follower_time'][1] - file['follower_time'][0])
        

        plt.plot(time_leader_joints, moving_average_leader_joints, linewidth= 2.0, label='reference trajecory')
        # plt.plot(actual_x, actual_y, linewidth=2.0, label='actual trajectory')
        # plt.xlabel('x axis (m)')
        # plt.ylabel('y axis (m)')
        # plt.legend()
        # plt.title('Reference vs actual trajectory when follower initially overshoots the goal')
        plt.show()
    
    if distractor == True:
        dist_time = file['distractor_times']
        print(dist_time[3] - dist_time[2])
        print(dist_time[1] - dist_time[0])
        print("Distractor_data: ", file['distractor_data'])
        print("Total_time: ", file['distractor_times'])

    # with_subplot_fn()
    # with_subplot_obstacle()