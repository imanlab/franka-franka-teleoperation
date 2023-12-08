#!/usr/bin/env python3

import rospy
import sys
import tf
import tf.transformations
import numpy as np
import copy
import tf2_ros
import time
from math import atan2, exp

from geometry_msgs.msg import Pose, PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path , Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool

from franka_msgs.msg import FrankaState
from sensor_msgs.msg import LaserScan, JointState
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal
import franka_gripper.msg
import control_msgs.msg 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import cv2

# Brings in the SimpleActionClient
import actionlib

# Switch between position and impedance control
from controller_manager_msgs.srv import SwitchController, ListControllers

# Moveit
import moveit_commander
import moveit_msgs.msg

import dynamic_reconfigure.client


class PandaToPandaTeleoperation:
    """ this class subscribe to joint states of leader and follower 
        and moves the follower arm related to leader arm.

        Optionally: 
            1) Provide haptics feedback 
            2) Provide virtual wall 
    """

    def __init__(self, leader_ns = "panda_leader", follower_ns = "panda_follower" ): 

        rospy.init_node("panda_panda_teleoperation") 

        self.leader_ns = leader_ns
        self.follower_ns = follower_ns
        
        rospy.loginfo("starting")
        self._teleoperation_variables()
        self._teleoperation_subscribers() 
        self._teleoperation_publishers() 
        # rospy.spin()
        self.control_loop()

    def _teleoperation_variables(self): 
 
        self.leader_joint_state = None
        self.follower_joint_state = None

        self.follower_joint_controller_pub = None
        self.leader_compliance_controller = None
        self.leader_compliance_config = None
        
    def _teleoperation_subscribers(self): 

        rospy.Subscriber( self.leader_ns + "/" + self.leader_ns + "_state_controller/franka_states" , FrankaState, self.leader_joint_state_cb )
        rospy.Subscriber( self.follower_ns + "/" + self.follower_ns + "_state_controller/franka_states" , FrankaState, self.follower_joint_state_cb )

    def _teleoperation_publishers(self): 

        self.follower_joint_controller_pub = rospy.Publisher( "/panda_follower/panda_follower_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )
        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )
        self.follower_arm_recovery_client.wait_for_server()

    def leader_joint_state_cb (self, msg ): 
        self.leader_joint_state = msg  
 
    def follower_joint_state_cb (self, msg ): 
        self.follower_joint_state = msg  
  
    def create_joint_trajectory(self): 

        traj_point = JointTrajectoryPoint()
        trajectory = JointTrajectory()
        
        follower_joint_names = []
        Kp = 1.3    #1.3
        Kd = 0.03   #0.03
        dt = 1.0    #1.75

        for i in range(7):
            follower_joint_name = self.follower_ns + "_joint" + str(i+1)
            follower_joint_names.append( follower_joint_name ) 

        leader_joint_values = self.leader_joint_state.q 
        follower_joint_values = self.follower_joint_state.q 
        # leader_joint_values = self.leader_joint_state.position[:7]
        # follower_joint_values = self.follower_joint_state.position[:7]


        position_error = [ i - j for i, j in zip(leader_joint_values, follower_joint_values ) ]
        velocity_error = self.leader_joint_state.dq  #self.leader_task_state.dq
    
        correction = [ Kp * p_err + Kd * v_err for p_err, v_err in zip( position_error, velocity_error ) ]
        traj_point.positions = [ i + j for i, j in zip( follower_joint_values , correction ) ]
        traj_point.velocities = velocity_error

        traj_point.time_from_start = rospy.Duration( dt )
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = follower_joint_names
        trajectory.points.append( traj_point )

        return trajectory

    def apply_leader_position_to_follower(self):
        trajectory = self.create_joint_trajectory()
        self.follower_joint_controller_pub.publish( trajectory )
 
    def recovery_follower_arm_error(self): 
        goal = ErrorRecoveryActionGoal() 
        self.follower_arm_recovery_client.send_goal(goal)
        # self.follower_arm_recovery_client.wait_for_result() 


    def control_loop(self): 
        
        move_gripper_to_leader_gripper = False

        rospy.loginfo("Press Ctrl+C if anything goes sideways")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown(): 
            try: 
                if  not isinstance( self.leader_joint_state , type(None) ) and \
                    not isinstance( self.follower_joint_state , type (None) ) :
                        self.apply_leader_position_to_follower()
                        self.recovery_follower_arm_error()
            except KeyboardInterrupt: 
                break

class checkTeleoperation:
    """ this class subscribe to joint states of leader and follower 
        and moves the follower arm related to leader arm.

        Optionally: 
            1) Provide haptics feedback 
            2) Provide virtual wall 
    """

    def __init__(self, leader_ns = "panda_leader", follower_ns = "panda_follower" ): 
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("check_panda_teleoperation") 

        self.leader_ns = leader_ns
        self.follower_ns = follower_ns

        self.user_name = sys.argv[1]
        self.condition = sys.argv[2]
        self.trial_number = sys.argv[3]
        
        rospy.loginfo("starting")
        self.listener = tf.TransformListener()
        self._teleoperation_variables()
        self._teleoperation_subscribers() 
        self._teleoperation_publishers() 
        # rospy.spin()

        # self.switch_controllers()
        self.control_loop()

    def _teleoperation_variables(self): 
        
        # Variables for follower arm teleoperation
        self.leader_joint_state = None
        self.follower_joint_state = None

        self.aruco_msg = None

        self.control_mobile_robot = False

        self.leader_stiff = True
        self.recorded_init_poses = False

        self.follower_joint_controller_pub = None  

        
        self.outsideWorkspace = False
        self.EElinkName = self.follower_ns + "_link8"

        self.data_written = False

        # Saving data
        self.save_reference_x = []
        self.save_reference_y = []
        self.save_reference_z = []
        self.save_actual_x = []
        self.save_actual_y = []
        self.save_actual_z = []

        self.save_mobile_reference_x = []
        self.save_mobile_reference_y = []
        self.save_mobile_actual_x = []
        self.save_mobile_actual_y = []

        self.save_mobile_haptic_feedback_x = []
        self.save_mobile_haptic_feedback_yaw = []

        self.save_haptic_feedback_x = []
        self.save_haptic_feedback_y = []
        self.save_haptic_feedback_z = []

        self.save_leader_joints = []
        self.save_follower_joints = []

        self.external_torques = []
        self.follower_external_torques = []

        # Variables for mobile robot teleoperation
        self.goalReached = False

        self.dz = 0.0
        self.high_external_torque = False

        self.currentLeaderFrankaPose = PoseStamped()

        self.virtualBoundaryTrans = [0.05, 0.4] # m
        self.virtualBoundaryWorkspace = 0.2 # m
        self.obstacle_speed_scale = 0.2
        self.speed_scale = 0.5
        
        self.withinWalls = False
        self.leader_stiff = False
        
        self.got_hunter_global_plan = False
        self.hunter_global_path = Path()

        self.initialized_hunter_orientation = False
        self.initial_hunter_orientation = 0

        self.too_close = False
        self.obstacle = False
        self.obstacle_haptic_force = []
        self.min_obstacle_distance = 1.75 # m

        self.got_waypoints = False
        
        self.control_follower_franka = False
        self.b_initialLeaderPoseFound = False
        self.initial_hunter_position_estimated = False

        self.goal_reached = False
        self.got_hunter_goal = False

        self.distractor = False

        self.initialLeaderFrankaPose = PoseStamped()

        # Time
        self.time_started = False
        self.follower_time_started = False
        self.start_time = 0
        self.end_time = 0

        self.start_follower_time = 0
        self.end_follower_time = 0

        self.dist = Bool()
        self.distractor_number = 0
        self.distractor = False
        self.distractors_collected = 0
        self.distractor_times = []
        self.all_distractors = []   # Array of size 2. First one is during teleop of mobile robot, second is teleop of arm

        # Moveit initialization
        # self.leader_commander = moveit_commander.RobotCommander(robot_description="/panda_leader/robot_description", ns="panda_leader")
        # self.leader_group = moveit_commander.MoveGroupCommander("panda_arm", robot_description="/panda_leader/robot_description",  ns="panda_leader")
        
    def _teleoperation_subscribers(self):
        # Subscribers for arm
        self.aruco_pose_sub = rospy.Subscriber( "/aruco_single/pose" , PoseStamped, self.aruco_object_cb )
        rospy.Subscriber( self.leader_ns + "/franka_state_controller/franka_states" , FrankaState, self.leader_franka_state_cb )
        rospy.Subscriber("/panda_leader/franka_state_controller/joint_states", JointState, self.leader_joint_state_cb)
        rospy.Subscriber("/panda_follower/panda_follower_state_controller/joint_states", JointState, self.follower_joint_state_cb)
        rospy.Subscriber( self.follower_ns + "/" + self.follower_ns + "_state_controller/franka_states" , FrankaState, self.follower_franka_state_cb )
        self.leaderComplinaceConfigServer = dynamic_reconfigure.client.Client(self.leader_ns + "/" + self.leader_ns +  "_cartesian_impedance_controller_controller_params", timeout=30, config_callback=self.compliance_config_cb)

        # Subscribers for hunter
        self.hunterNavGoalSub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.hunter_nav_goal_cb)
        self.hunterVisualPathSub = rospy.Subscriber("/move_base/DWAPlannerROS/global_plan", Path, self.hunter_nav_path_cb)
        self.lidarSub = rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        self.hunterOdomSub = rospy.Subscriber("/odom", Odometry, self.hunter_odom_cb)
        
    def _teleoperation_publishers(self): 
        self.pub_follower_control = rospy.Publisher("/follower_control", Bool, queue_size=5)
        self.pub_end_task = rospy.Publisher("/end_task", Bool, queue_size=5)

        self.leader_joint_controller_pub = rospy.Publisher( "/panda_leader/position_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )
        self.follower_joint_controller_pub = rospy.Publisher( "/panda_follower/panda_follower_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )
        self.leaderFrankaRecoveryPub = rospy.Publisher(self.leader_ns + '/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10)
        
        # Publishers for hunter
        self.initialHunterPosePub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
        self.HunterGoalPub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.hunterCmdVelPub = rospy.Publisher( "/cmd_vel", Twist, queue_size=10)

        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )
        self.follower_arm_recovery_client.wait_for_server()

    def compliance_config_cb(self, config):
        self.leaderComplianceParams = config

    def aruco_object_cb (self, msg ): 

        self.aruco_msg = msg
    
    def follower_franka_state_cb (self, msg ): 
        self.follower_joint_state = msg
        if self.control_follower_franka:
            self.save_follower_joints.append(msg.q)  

    def follower_joint_state_cb(self, msg):
        self.follower_curr_joint_states = msg

    def leader_joint_state_cb(self, msg):
        self.leader_curr_joint_states = msg

    def leader_franka_state_cb(self, msg):
        """
        This function defines the intial pose the franka robot is present in, as the reference.
        Next, the haptic feedback functions are called
        """
        self.leader_joint_state = msg
        if self.control_follower_franka:
            self.save_leader_joints.append(msg.q)

    def franka_leader_main(self):
        if self.control_follower_franka == True:
            self.dist.data = True
            self.pub_follower_control.publish(self.dist)
            # Get leader pose in Euler angles
            leader_current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(self.leader_joint_state.O_T_EE, (4, 4))))
            leader_current_quaternion = leader_current_quaternion / np.linalg.norm(leader_current_quaternion)

            leader_pose = PoseStamped()
            leader_pose.header.frame_id = self.EElinkName
            leader_pose.header.stamp = rospy.Time(0)
            leader_pose.pose.orientation.x = leader_current_quaternion[0]
            leader_pose.pose.orientation.y = leader_current_quaternion[1]
            leader_pose.pose.orientation.z = leader_current_quaternion[2]
            leader_pose.pose.orientation.w = leader_current_quaternion[3]
            leader_pose.pose.position.x = self.leader_joint_state.O_T_EE[12]
            leader_pose.pose.position.y = self.leader_joint_state.O_T_EE[13]
            leader_pose.pose.position.z = self.leader_joint_state.O_T_EE[14]

            # Get follower pose in Euler angles
            follower_current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(self.follower_joint_state.O_T_EE, (4, 4))))
            follower_current_quaternion = follower_current_quaternion / np.linalg.norm(follower_current_quaternion)

            follower_pose = PoseStamped()
            follower_pose.header.frame_id = self.EElinkName
            follower_pose.header.stamp = rospy.Time(0)
            follower_pose.pose.orientation.x = follower_current_quaternion[0]
            follower_pose.pose.orientation.y = follower_current_quaternion[1]
            follower_pose.pose.orientation.z = follower_current_quaternion[2]
            follower_pose.pose.orientation.w = follower_current_quaternion[3]
            follower_pose.pose.position.x = self.follower_joint_state.O_T_EE[12]
            follower_pose.pose.position.y = self.follower_joint_state.O_T_EE[13]
            follower_pose.pose.position.z = self.follower_joint_state.O_T_EE[14]

            if not self.recorded_init_poses:
                # print("Not recorded init poses")

                # Stop the mobile robot and record end times
                desired_cmd_vel = Twist()
                desired_cmd_vel.linear.x = 0
                desired_cmd_vel.angular.z = 0
                self.hunterCmdVelPub.publish( desired_cmd_vel )
                print("=========  CONGRATULATIONS - OBJECT DETECTED ================")
                self.end_time = rospy.get_time()
                self.goalReached = True

                # self.initial_position(leader_pose)
                self.recovery_leader_arm_error()
                self.recovery_follower_arm_error()
                self.switch_controllers("position_joint_trajectory_controller", "panda_leader_cartesian_impedance_controller")
                self.homing(5)
                self.franka_leader_init = leader_pose
                self.franka_follower_init = follower_pose
                self.recorded_init_poses = True
                self.outsideWorkspace = False
                # self.all_distractors.append(self.distractors_collected)
                self.distractor_number = 0
                self.distractors_collected = 0
                print("========= YOU CAN NOW CONTROL THE FOLLOWER ARM =======================")
                self.start_follower_time = rospy.get_time()

            if self.leader_stiff == True:
                self.recovery_leader_arm_error()
                self.recovery_follower_arm_error()
                name, state = self.list_controllers("position_joint_trajectory_controller")
                if state == "running":
                    self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
                    print("Switching to cartesian impedamnce control")
                # elif state == "initialized" or state == "stopped":
                self.leaderComplianceParams['translational_x_stiffness'] = 0
                self.leaderComplianceParams['translational_y_stiffness'] = 0
                self.leaderComplianceParams['translational_z_stiffness'] = 0
                self.leaderComplianceParams['rotational_x_stiffness'] = 0
                self.leaderComplianceParams['rotational_y_stiffness'] = 0
                self.leaderComplianceParams['rotational_z_stiffness'] = 0
                self.leaderComplianceParams['task_haptic_x_force'] = 0
                self.leaderComplianceParams['task_haptic_y_force'] = 0
                self.leaderComplianceParams['task_haptic_z_force'] = 0
                self.leaderComplianceParams['task_haptic_z_torque'] = 0
                self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)
                self.leader_stiff = False  

            follower_external_torque = self.leader_joint_state.tau_ext_hat_filtered

            if not self.goal_reached:
                self.follower_external_torques.append(follower_external_torque[6])

                # Check for torque due to external forces on EE in Nm
                # Currently not doing anything. Only checking for high external torques
                if abs(follower_external_torque[6]) >= 1.0:
                    self.high_external_torque = True
                else:
                    self.high_external_torque = False

                trajectory = self.mimic_joint_trajectory()
                self.follower_joint_controller_pub.publish( trajectory )

                planned_path = self.generate_trajectory()
                self.normal_landing_point(follower_pose, planned_path)

                # if self.distractor == False:
                # self.distractors()


            elif self.goal_reached:
                self.end_follower_time = rospy.get_time()
                # self.all_distractors.append(self.distractors_collected)
                self.write_to_file()


        if self.control_follower_franka == False:

            self.dist.data = False
            self.pub_follower_control.publish(self.dist)


            current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(self.leader_joint_state.O_T_EE, (4, 4))))
            current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)

            leader_pose = PoseStamped()
            leader_pose.header.frame_id = self.EElinkName
            leader_pose.header.stamp = rospy.Time(0)
            leader_pose.pose.orientation.x = current_quaternion[0]
            leader_pose.pose.orientation.y = current_quaternion[1]
            leader_pose.pose.orientation.z = current_quaternion[2]
            leader_pose.pose.orientation.w = current_quaternion[3]
            leader_pose.pose.position.x = self.leader_joint_state.O_T_EE[12]
            leader_pose.pose.position.y = self.leader_joint_state.O_T_EE[13]
            leader_pose.pose.position.z = self.leader_joint_state.O_T_EE[14]
            # Initializing everything
            if not (self.b_initialLeaderPoseFound):
                self.recovery_leader_arm_error()
                self.recovery_follower_arm_error()
                self.homing(5)

                hunter_goalPose = PoseStamped()
                hunter_goalPose.header.frame_id = "map"
                hunter_goalPose.pose.position.x = 0.3157888352870941
                hunter_goalPose.pose.position.y = -2.953887939453125
                hunter_goalPose.pose.position.z = 0.0
                hunter_goalPose.pose.orientation.x = 0.0
                hunter_goalPose.pose.orientation.y = 0.0
                hunter_goalPose.pose.orientation.z = -0.45827655191204114
                hunter_goalPose.pose.orientation.w = 0.8888096545198023
                # self.HunterGoalPub.publish(hunter_goalPose)

                if not self.initial_hunter_position_estimated:
                    self.initial_hunter_position()
                    self.initial_hunter_position_estimated = True
                self.initialLeaderFrankaPose = leader_pose
                # print("Initial leader franka pose: ", self.initialLeaderFrankaPose)
                self.b_initialLeaderPoseFound = True
                self.distractor_number = 0
                print("========= YOU CAN NOW CONTROL THE MOBILE ROBOT=======================")
            else: 
                self.currentLeaderFrankaPose = leader_pose

            if self.leader_stiff == False:
                self.recovery_leader_arm_error()
                self.recovery_follower_arm_error()
                name, state = self.list_controllers("position_joint_trajectory_controller")
                if state == "running":
                    self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
                    print("Switched to cartesian impedance control")   
                # elif state == "initialized" or state == "stopped":
                self.leaderComplianceParams['translational_x_stiffness'] = 0
                self.leaderComplianceParams['translational_y_stiffness'] = 400
                self.leaderComplianceParams['translational_z_stiffness'] = 400
                self.leaderComplianceParams['rotational_x_stiffness'] = 30
                self.leaderComplianceParams['rotational_y_stiffness'] = 30
                self.leaderComplianceParams['rotational_z_stiffness'] = 0
                self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)
                self.leader_stiff = True

            external_torque = self.leader_joint_state.tau_ext_hat_filtered
            
            # Record torques at EE when the goal is defined but not reached
            if (self.got_hunter_goal) and (not self.goalReached):
                self.external_torques.append(external_torque[6])
                # self.distractors()

            # Check for torque due to external forces on EE in Nm
            if abs(external_torque[6]) >= 1.0:
                self.high_external_torque = True
                self.leader_motion() # haptic navigation
            else:
                self.high_external_torque = False
                self.leader_motion() # haptic navigation

            # if self.distractor == False:
            

    # def distractors(self):      

    #     if self.control_follower_franka == False:
    #         if self.distractor_number < 4 and (rospy.get_time() >= self.start_time + self.distractor_times[self.distractor_number]) \
    #             and (rospy.get_time() <= self.start_time + self.distractor_times[self.distractor_number+1]):
    #             self.distractor_number += 1
    #             self.distractor_start_time = time.time()
    #             self.distractor = True
    #             self.distractor_display_collect()

    #         elif self.distractor_number == 4:
    #             self.distractor_number += 1
    #             self.distractor = True
    #             self.distractor_display_collect()

    #         elif self.distractor == True:
    #             self.distractor_display_collect()

    #     elif self.control_follower_franka:
    #         # print("All distractor times: ", self.distractor_times)
    #         # print("Current time: ", rospy.get_time())
    #         # print("Start time: ", self.start_follower_time)
    #         # print("Distractor time: ", self.distractor_times[self.distractor_number])
    #         # print("Compare: ", self.start_follower_time + self.distractor_times[self.distractor_number])
    #         if self.distractor_number < 4 and (rospy.get_time() >= self.start_follower_time + self.distractor_times[self.distractor_number]) \
    #             and (rospy.get_time() <= self.start_follower_time + self.distractor_times[self.distractor_number+1]):
    #             print(" ")
    #             print("Showing distractor")
    #             self.distractor_number += 1
    #             self.distractor_start_time = time.time()
    #             self.distractor = True
    #             self.distractor_display_collect()

    #         elif self.distractor_number == 4:
    #             self.distractor_number += 1
    #             self.distractor = True
    #             self.distractor_display_collect()

    #         elif self.distractor == True:
    #             self.distractor_display_collect()
            
        

    # def distractor_display_collect(self):
        
        
    #     if self.distractor == True:
    #         print("In display distractor")
    #         image = np.zeros(  ( 200, 200, 3 ) , dtype = np.uint8)
    #         image[:,:, 2] = 255
    #         cv2.imshow( "distractor", image )
    #         k = cv2.waitKey(30)
    #         if (time.time() - self.distractor_start_time) > 1:
    #             cv2.destroyWindow("distractor")
    #             self.distractor = False
    #         elif k == ord(' ') or k == 32:
    #             self.distractors_collected += 1
    #             print("collected: ", self.distractors_collected)
    #             cv2.destroyWindow("distractor")
    #             self.distractor = False
    #         elif k < 0:
    #             pass

            
    # def get_times(self):
    #     time1 = np.random.randint(4,10)
    #     time2 = np.random.randint(13, 20)
    #     time3 = np.random.randint(23, 30)
    #     time4 = np.random.randint(33, 40)
    #     time5 = np.random.randint(43, 50)

    #     self.distractor_times = [time1, time2, time3, time4, time5]



    def home_trajectory(self):

        home_pose = PoseStamped()

        home_pose.pose.position.x = 0.37317982974158986
        home_pose.pose.position.y = 0.018578472172744866
        home_pose.pose.position.z = 0.37742714575804354

        home_pose.pose.orientation.x = 0.9997454239929339
        home_pose.pose.orientation.y = 0.01325016111341874
        home_pose.pose.orientation.z = -0.008546640621736785
        home_pose.pose.orientation.w = 0.01613924935492327

        return home_pose

    # def homing(self, T):
    #     # Moveit code
    #     # joint_goal = self.leader_group.get_current_joint_values()
    #     # # print(self.leader_commander.get_current_state())
    #     # joint_goal[0] = -0.12
    #     # joint_goal[1] = -0.45
    #     # joint_goal[2] = -0.06
    #     # joint_goal[3] = -2.39
    #     # joint_goal[4] = -0.02
    #     # joint_goal[5] = 1.96
    #     # joint_goal[6] = 0.85
    #     print ( self.leader_group.get_current_pose() )
    #     self.leader_group.set_named_target("leader_home")
    #     self.leader_group.go(wait = True ) 
    #     time.sleep(1)
    #     # print("Am here")
    #     # self.leader_group.go(joint_goal, wait=True)
    #     # print("completed waiting")
    
    def homing(self, time_to_reach_home):
        time_to_reach_home = 5.0 # sec
        self.recovery_leader_arm_error()
        self.recovery_follower_arm_error()
        name, state = self.list_controllers("position_joint_trajectory_controller")
        if state == "initialized" or state == "stopped":
            self.switch_controllers("position_joint_trajectory_controller", "panda_leader_cartesian_impedance_controller")
        
        # self.recovery_leader_arm_error()
        # self.recovery_follower_arm_error()


        print("Homing")

        leader_joint_values = self.leader_curr_joint_states.position
        current_vals = JointTrajectoryPoint()
        current_vals.positions = self.leader_joint_state.q
        # current_vals.positions = [leader_joint_values[0], leader_joint_values[1], leader_joint_values[2],leader_joint_values[3],\
        #  leader_joint_values[4], leader_joint_values[5], leader_joint_values[6]]
        
        # 7 joint vals at home position
        joint_home = JointTrajectoryPoint()
        joint_home.positions = [0.007481502518816376, -0.5204791008017913, -0.04640116417722879, -2.272347191082633, -0.012951721175211418, 1.811533434867859, 0.8056091825067997]
        joint_home.time_from_start = rospy.Time(time_to_reach_home)

        traj = JointTrajectory()
        leader_joint_names = []
        for i in range(7):
            leader_joint_name = "panda"+ "_joint" + str(i+1)
            leader_joint_names.append( leader_joint_name ) 
        traj.joint_names = leader_joint_names
        traj.points = [current_vals, joint_home]

        r = rospy.Rate(1) # 10hz
        self.leader_joint_controller_pub.publish(traj)
        r.sleep()
        self.leader_joint_controller_pub.publish(traj)
        # time.sleep(time_to_reach_home + 0.5) # Code stops for time_to_reach + 0.5 seconds, hence waiting for traj to finish publishing
        # if (leader_joint_values[0] - joint_home.positions[0] < 0.02) and (leader_joint_values[1] - joint_home.positions[1] < 0.02) \
        # and (leader_joint_values[2] - joint_home.positions[2] < 0.02) and (leader_joint_values[3] - joint_home.positions[3] < 0.02) \
        # and (leader_joint_values[4] - joint_home.positions[4] < 0.02) and (leader_joint_values[5] - joint_home.positions[5] < 0.02) \
        # and (leader_joint_values[6] - joint_home.positions[6] < 0.02):
        #     print("Homing Done")
        #     self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")

        if (leader_joint_values[0] - joint_home.positions[0] < 0.06) and (leader_joint_values[1] - joint_home.positions[1] < 0.06) \
        and (leader_joint_values[2] - joint_home.positions[2] < 0.06) and (leader_joint_values[3] - joint_home.positions[3] < 0.06) \
        and (leader_joint_values[4] - joint_home.positions[4] < 0.06) and (leader_joint_values[5] - joint_home.positions[5] < 0.06) \
        and (leader_joint_values[6] - joint_home.positions[6] < 0.06):
            # print("Homing close")
            if self.control_follower_franka == False:
                self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")   
            else:
                rospy.sleep(3)
                print("Slept for 3 seconds")
                self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
                print("========= YOU CAN NOW CONTROL THE ROBOT=======================")   
        else:
            self.homing(5)
        
    def home_path(self):
        if not self.got_waypoints:
            self.waypoints()


        leader_joint_names = []

        for i in range(7):
            leader_joint_name = "panda"+ "_joint" + str(i+1)
            leader_joint_names.append( leader_joint_name ) 

        # time_range = np.arange(0, end_time, end_time/num_steps)
        time_range = [0,6]
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = leader_joint_names
        # print("In func")
        for time, way_points in zip(time_range, self.total_path):
            traj_point = JointTrajectoryPoint()
            traj_point.time_from_start = rospy.Duration(time)
            traj_point.positions = way_points
            trajectory.points.append( traj_point )
        self.leader_joint_controller_pub.publish( trajectory )
        # rospy.sleep(end_time)

        # diff_joint0 = (leader_joint_values[0] - joint_home[0])/num_steps
        # diff_joint1 = (leader_joint_values[1] - joint_home[1])/num_steps
        # diff_joint2 = (leader_joint_values[2] - joint_home[2])/num_steps
        # diff_joint3 = (leader_joint_values[3] - joint_home[3])/num_steps
        # diff_joint4 = (leader_joint_values[4] - joint_home[4])/num_steps
        # diff_joint5 = (leader_joint_values[5] - joint_home[5])/num_steps
        # diff_joint6 = (leader_joint_values[6] - joint_home[6])/num_steps

        # for i in range(0, num_steps):
        #     joint_0 = leader_joint_values[0] + diff_joint0 * i
        #     joint_1 = leader_joint_values[1] + diff_joint1 * i
        #     joint_2 = leader_joint_values[2] + diff_joint2 * i
        #     joint_3 = leader_joint_values[3] + diff_joint3 * i
        #     joint_4 = leader_joint_values[4] + diff_joint4 * i
        #     joint_5 = leader_joint_values[5] + diff_joint5 * i
        #     joint_6 = leader_joint_values[6] + diff_joint6 * i
            # self.total_path.append([joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]

        # for steps in range(num_steps):
        #     path_vals = [i-j for i, j in zip(leader_joint_values, joint_home)]
        # path_vals = leader_joint_values
        # print("Path Vals: ", path_vals)
        # home_pose_check = self.home_trajectory()
        # print("X: ", self.leader_joint_state.O_T_EE[12])
        # print("home X: ", home_pose_check.pose.position.x)

        # if (self.leader_joint_state.O_T_EE[12] - home_pose_check.pose.position.x) < 0.02 and (self.leader_joint_state.O_T_EE[13] - home_pose_check.pose.position.y < 0.02):
        #     print("Reached home position")
        #     self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
        #     # rospy.sleep(10)
        #     # Put switch controller here
        # else:
        #     print("Homing")
        #     self.home_path()

        # while not ((self.leader_joint_state.O_T_EE[12] - home_pose_check.pose.position.x) < 0.02 and (self.leader_joint_state.O_T_EE[13] - home_pose_check.pose.position.y < 0.02)):
        #     rospy.sleep(0.1)
        print("Reached home position") 

    def mimic_joint_trajectory(self): 
        traj_point = JointTrajectoryPoint()
        trajectory = JointTrajectory()
        
        follower_joint_names = []
        Kp = 1.3    #1.3
        Kd = 0.03   #0.03
        dt = 1.0    #1.75

        for i in range(7):
            follower_joint_name = self.follower_ns + "_joint" + str(i+1)
            follower_joint_names.append( follower_joint_name ) 

        leader_joint_values = self.leader_joint_state.q 
        follower_joint_values = self.follower_joint_state.q 
        # leader_joint_values = self.leader_joint_state.position[:7]
        # follower_joint_values = self.follower_joint_state.position[:7]


        position_error = [ i - j for i, j in zip(leader_joint_values, follower_joint_values ) ]
        velocity_error = self.leader_joint_state.dq  #self.leader_task_state.dq
    
        correction = [ Kp * p_err + Kd * v_err for p_err, v_err in zip( position_error, velocity_error ) ]
        traj_point.positions = [ i + j for i, j in zip( follower_joint_values , correction ) ]
        # print("traj_point: ", traj_point)
        traj_point.velocities = velocity_error

        traj_point.time_from_start = rospy.Duration( dt )
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = follower_joint_names
        trajectory.points.append( traj_point )

        return trajectory

    def record_data(self, position_vals_type, position_vals_array):

        # Save reference and actual positions of End Effector
        if len(position_vals_array) == 0:
            pass
        elif len(position_vals_array) > 0:
            if position_vals_type == "follower":
                self.save_reference_x.append(position_vals_array[3])
                self.save_reference_y.append(position_vals_array[4])
                self.save_reference_z.append(position_vals_array[5])
                self.save_actual_x.append(position_vals_array[0])
                self.save_actual_y.append(position_vals_array[1])
                self.save_actual_z.append(position_vals_array[2])
            elif position_vals_type == "hunter":
                self.save_mobile_reference_x.append(position_vals_array[2])
                self.save_mobile_reference_y.append(position_vals_array[3])
                self.save_mobile_actual_x.append(position_vals_array[0])
                self.save_mobile_actual_y.append(position_vals_array[1])
                
        # Record torques at EE when the goal is defined but not reached
        # if (self.got_thorvald_goal) and (not self.goalReached):
        #     self.external_torques.append(external_torque[6])




    def object_pose(self):
        """
        Gets the aruco marker pose from camera's optical frame to the panda's base link frame 
        """
        object_pose = PoseStamped()

        self.listener.waitForTransform("/" + self.follower_ns + "_link0", "/camera_color_optical_frame",rospy.Time(), rospy.Duration(1.0))
        (trans, rot) = self.listener.lookupTransform("/" + self.follower_ns + "_link0", "/camera_color_optical_frame", rospy.Time())

        if not isinstance(self.aruco_msg, type(None) ):     # Make sure that a the aruco marker is detected
            object_pose.pose.position.x = self.aruco_msg.pose.position.x * trans[0]
            object_pose.pose.position.y = self.aruco_msg.pose.position.y * trans[1]
            object_pose.pose.position.z = self.aruco_msg.pose.position.z * trans[2]

            object_pose.pose.orientation.x = self.aruco_msg.pose.orientation.x * rot[0]
            object_pose.pose.orientation.y = self.aruco_msg.pose.orientation.y * rot[1]
            object_pose.pose.orientation.z = self.aruco_msg.pose.orientation.z * rot[2]
            object_pose.pose.orientation.w = self.aruco_msg.pose.orientation.w * rot[3]

        return object_pose
               
        
    def generate_trajectory(self):
        numPoints = 200
        if not self.recorded_init_poses:
            goal_state = self.home_trajectory()
            joint_state = self.leader_joint_state
        else:
            goal_state = self.object_pose()
            joint_state = self.follower_joint_state

        current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(joint_state.O_T_EE, (4, 4))))
        current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)

        EE_pose = PoseStamped()
        EE_pose.header.frame_id = self.EElinkName
        EE_pose.header.stamp = rospy.Time(0)
        EE_pose.pose.orientation.x = current_quaternion[0]
        EE_pose.pose.orientation.y = current_quaternion[1]
        EE_pose.pose.orientation.z = current_quaternion[2]
        EE_pose.pose.orientation.w = current_quaternion[3]
        EE_pose.pose.position.x = joint_state.O_T_EE[12]
        EE_pose.pose.position.y = joint_state.O_T_EE[13]
        EE_pose.pose.position.z = joint_state.O_T_EE[14]
        t = time.time()
        x_current, y_current, z_current = EE_pose.pose.position.x, EE_pose.pose.position.y, EE_pose.pose.position.z
        x_goal, y_goal, z_goal = goal_state.pose.position.x, goal_state.pose.position.y, goal_state.pose.position.z
        dx, dy, dz = (x_goal - x_current) / numPoints, (y_goal - y_current) / numPoints, ( z_goal - z_current ) / numPoints

        path = Path()
        path.header.stamp = rospy.Time().now()
        path.header.frame_id = "odom"
        path.poses = []

        for i in range(numPoints):
            cart_pose = PoseStamped()
            cart_pose.pose.position.x = x_current + dx * i
            cart_pose.pose.position.y = y_current + dy * i
            cart_pose.pose.position.z = z_current + dz * i
            cart_pose.pose.orientation.x = 0.0
            cart_pose.pose.orientation.y = 0.0
            cart_pose.pose.orientation.z = 0.0
            cart_pose.pose.orientation.w = 1.0 

            path.poses.append( cart_pose ) 
        return path

    
    def normal_landing_point(self, point, plan):
        """
        Landing point is the closest point on the planned trajectory in front of the robot's current position that can be reached smoothly
        """
        
        minEulerDistance = np.Inf
        follower_position_vals = []
        hunter_position_vals = []
        obj_goal = self.object_pose()

        for i in range( len(plan.poses) ):
            if self.control_follower_franka == True:
                x1, y1, z1 = point.pose.position.x , point.pose.position.y, point.pose.position.z  # Franka EE's current coordinates
                x2, y2, z2 = plan.poses[i].pose.position.x, plan.poses[i].pose.position.y, plan.poses[i].pose.position.z   # The planned trajectory's coordinates      

                # Check all points on the generated path that are ahead (close to object) of the Franka EE base
                if z2 < z1:     # ("Goal below the arm")
                    dist = np.sqrt( np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1))
                    if dist < minEulerDistance:
                        minEulerDistance = dist
                        try:
                            # print("Plan is: ", plan)
                            franka_orientation = point.pose.orientation
                            landing_point = plan.poses[i+20].pose.orientation
                            landing_point_pos = plan.poses[i+20].pose.position
                        except IndexError:
                            # The goal is in front of the robot but closer than 20 steps ahead
                            franka_orientation = point.pose.orientation
                            landing_point = plan.poses[i].pose.orientation
                            landing_point_pos = plan.poses[i].pose.position

                    #  The following is to save reference trajectory in order to compare with traj followed bu user
                    follower_position_vals.append([x1, y1, z1, landing_point_pos.x, landing_point_pos.y, landing_point_pos.z])
                else:
                    follower_position_vals.append([x1, y1, z1, obj_goal.pose.position.x, obj_goal.pose.position.y, obj_goal.pose.position.z])



            elif self.control_follower_franka == False:
                x1, y1 = point.pose.position.x , point.pose.position.y  # Hunter's base coordinates
                x2, y2 = plan.poses[i].pose.position.x, plan.poses[i].pose.position.y   # The planned trajectory's coordinates 
                
                # Check all points on the generated path that are ahead of the robot's base
                if x2 > x1:     # print("Goal ahead")
                    dist = np.sqrt( np.square(x2 - x1) + np.square(y2 - y1) )
                    if dist < minEulerDistance:
                        minEulerDistance = dist

                        # Get dYaw based on the landing point and the thorvald orientation
                        try:
                            hunter_orientation = point.pose.orientation
                            landing_point = plan.poses[i+40].pose.orientation
                            landing_point_pos = plan.poses[i+40].pose.position
                            dYaw, heading = self.get_dYaw(hunter_orientation, landing_point)
                        except IndexError:
                            # The goal is in front of the robot but closer than 40 steps ahead
                            self.too_close = True
                            hunter_orientation = point.pose.orientation
                            landing_point = plan.poses[i].pose.orientation
                            landing_point_pos = plan.poses[i].pose.position
                            dYaw, heading = self.get_dYaw(hunter_orientation, landing_point)
                        
                        hunter_position_vals.append([x1, y1, landing_point_pos.x, landing_point_pos.y])
                # If the robot is front of goal then the landing point is the goal
                else:
                    hunter_orientation = point.pose.orientation
                    landing_point = self.goal.pose.orientation
                    landing_point_pos = self.goal.pose.position
                    dYaw, heading = self.get_dYaw(hunter_orientation, landing_point)
                    # Landing point x, y are the goal's x and y
                    x2 = self.goal.pose.position.x
                    y2 = self.goal.pose.position.y

                    hunter_position_vals.append([x1, y1, x2, y2])
                # dist_x = abs(x2 - x1) - self.franka_follower_init.pose.x
                # dist_y = abs(y2 - y1) - self.franka_follower_init.pose.y
                # if dist_x > dist_y:

        if self.control_follower_franka == True:
            # print("Follower position vals", follower_position_vals)
            follower_position_vals = follower_position_vals[-1]
            self.record_data(position_vals_type="follower", position_vals_array= follower_position_vals)

            self.leader_arm_haptic_feedback(follower_position_vals)
        else:
            hunter_position_vals = hunter_position_vals[-1]
            self.record_data(position_vals_type="hunter", position_vals_array = hunter_position_vals)

            self.leader_mobile_haptic_feedback(hunter_position_vals, dYaw, heading)
        

    def leader_arm_haptic_feedback(self, position_vals):
        # if ((self.goalReached) and (self.got_thorvald_global_plan)) or (self.too_close):
        #     self.leaderComplianceParams['task_haptic_x_force'] = 0
        #     self.leaderComplianceParams['task_haptic_z_torque'] = 0
        #     self.too_close = False
        # else:

        # Goal is below the robot's current EE postion
        if position_vals[5] < position_vals[2]:

            if position_vals[0] < position_vals[3]:
                # print("0: ", position_vals[0])
                # print("3: ", position_vals[3])
                self.leaderComplianceParams['task_haptic_x_force'] = -4    # Move back towards base
            else:
                # print("Far")
                # print("0: ", position_vals[0])
                # print("3: ", position_vals[3])
                self.leaderComplianceParams['task_haptic_x_force'] = 4   # Move forward away from base
            # elif dist_y > dist_x:
            if position_vals[1] < position_vals[4]:
                self.leaderComplianceParams['task_haptic_y_force'] = 2    # Move to right
            else:
                self.leaderComplianceParams['task_haptic_y_force'] = -2   # Move to left
            self.leaderComplianceParams['task_haptic_z_force'] = -3   # Move down
        # If goal is above the robot's curent EE postion
        else:
            self.leaderComplianceParams['task_haptic_x_force'] = 0
            self.leaderComplianceParams['task_haptic_y_force'] = 0
            self.leaderComplianceParams['task_haptic_z_force'] = 3   # Move up

        self.save_haptic_feedback_x.append(self.leaderComplianceParams['task_haptic_x_force'])
        self.save_haptic_feedback_y.append(self.leaderComplianceParams['task_haptic_y_force'])
        self.save_haptic_feedback_z.append(self.leaderComplianceParams['task_haptic_z_force'])

        if self.condition == "h" or self.condition == "vh":
            self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)
        

    def recovery_follower_arm_error(self): 
        goal = ErrorRecoveryActionGoal() 
        self.follower_arm_recovery_client.send_goal(goal)
        # self.follower_arm_recovery_client.wait_for_result() 

    def recovery_leader_arm_error(self): 
        # recover leader franka 
        error_msg = ErrorRecoveryActionGoal()
        self.leaderFrankaRecoveryPub.publish( error_msg ) 


    """
    The following code controls the mobile robot
    """
    def hunter_nav_goal_cb(self, msg):
        """
        This function gets the pose of the 2d navigation point set by the user
        """
        if isinstance(msg.pose.position.x, float):
            self.goal = msg
            self.got_hunter_goal = True

    def hunter_nav_path_cb(self, msg):
        """
        This function gets the path to goal generated by the global planner
        """
        if (self.got_hunter_goal) and (not self.got_hunter_global_plan):
            print("Got Hunter plan")            
            self.got_hunter_global_plan = True
            self.initialized_hunter_orientation = True

        self.hunter_global_path = msg

    def lidar_cb(self, msg):
        # This function takes in lidar values as input and provides haptic_feedback when obstacle is very close
        self.lidar_ranges = msg.ranges

    def hunter_odom_cb(self, msg):
        """
        This function adjusts the stiffness of the franka robot based on the current position of Franka
        TODO: Check if the value of 0.2 is fine. Initially was 0.03
        """
        # offset_x = abs(msg.pose.pose.position.x - self.goal.pose.position.x) - 2.1709985733
        # offset_y = abs(msg.pose.pose.position.y - self.goal.pose.position.y) - 0.539679527283
        self.hunter_odom_msg = msg


    def initial_hunter_position(self):

        hunter_initpose = PoseWithCovarianceStamped()
        hunter_initpose.header.frame_id = "map"
        hunter_initpose.pose.pose.position.x = -3.2796120643615723
        hunter_initpose.pose.pose.position.y = 5.120046615600586
        hunter_initpose.pose.pose.position.z = 0.0
        hunter_initpose.pose.pose.orientation.x = 0.0
        hunter_initpose.pose.pose.orientation.y = 0.0
        hunter_initpose.pose.pose.orientation.z = -0.7024586598450735
        hunter_initpose.pose.pose.orientation.w = 0.7117245472854392
        self.initialHunterPosePub.publish(hunter_initpose)

        desired_cmd_vel = Twist()
        desired_cmd_vel.linear.x = 0
        desired_cmd_vel.angular.z = 0
        self.hunterCmdVelPub.publish( desired_cmd_vel )


    def leader_mobile_haptic_feedback(self, curr_close_positions, dYaw, heading):
        """
        This functions takes into account the path to the goal position and gives feedback on the leader arm
        """
        if not self.control_follower_franka:
            if ((self.goalReached) and (self.got_hunter_global_plan)) or (self.too_close):
                self.leaderComplianceParams['task_haptic_x_force'] = 0
                self.leaderComplianceParams['task_haptic_z_torque'] = 0
                self.too_close = False

            # Haptic force pushing backward when obstacle is in front
            # elif self.obstacle:
            #     self.leaderComplianceParams['task_haptic_x_force'] = 5 * np.mean(self.lidar_ranges)  
            #     self.leaderComplianceParams['task_haptic_z_torque'] = 0

            else:
                # If mobile robot is ahead of goal and the robot is facing south, then haptic force applied towards leader arm base
                if (curr_close_positions[2] - curr_close_positions[0] < 0) and (heading > 2.5):
                    if abs(50 * (curr_close_positions[2] - curr_close_positions[0])) >= 6:
                        self.leaderComplianceParams['task_haptic_x_force'] = -6
                    else:
                        self.leaderComplianceParams['task_haptic_x_force'] = -abs(50 * (curr_close_positions[2] - curr_close_positions[0]))
                # If mobile robot is ahead of goal, then haptic force applied away from leader arm base
                elif curr_close_positions[2] - curr_close_positions[0] < 0:
                    if abs(50 * (curr_close_positions[2] - curr_close_positions[0])) >= 6:
                        self.leaderComplianceParams['task_haptic_x_force'] = 6
                    else:
                        self.leaderComplianceParams['task_haptic_x_force'] = abs(50 * (curr_close_positions[2] - curr_close_positions[0]))

                # Two cases when robot is behind goal
                # If mobile robot is behind goal and robot is facing south, then haptic force applied away from leader arm base
                elif (curr_close_positions[2] - curr_close_positions[0] > 0) and (heading > 2.5):
                    if abs(50 * (curr_close_positions[2] - curr_close_positions[0])) >= 6:
                        self.leaderComplianceParams['task_haptic_x_force'] = 6
                    else:
                        self.leaderComplianceParams['task_haptic_x_force'] = abs(50 * (curr_close_positions[2] - curr_close_positions[0]))
                # If mobile robot is behind goal, then haptic force applied towards leader arm base
                elif curr_close_positions[2] - curr_close_positions[0] > 0:
                    if abs(50 * (curr_close_positions[2] - curr_close_positions[0])) >= 6:
                        self.leaderComplianceParams['task_haptic_x_force'] = -6
                    else:
                        self.leaderComplianceParams['task_haptic_x_force'] = -abs(50 * (curr_close_positions[2] - curr_close_positions[0]))

                #If dYaw is -ve (goal is to the right of robot), turn thorvald right 
                # i.e task_haptic_z_torque is +ve (EE turns CCW i.e towards franka A)
                
                # else:
                # If an external torque of more than 1Nm is applied, then the feedback decreases exponentially
                if self.high_external_torque:
                    # print("High external")
                    self.leaderComplianceParams['task_haptic_z_torque'] = 1 / (3*exp(abs(dYaw)))
                elif abs(self.dz) >= 0.5:
                    # print("dz > 0.5")
                    self.leaderComplianceParams['task_haptic_z_torque'] = 0
                else:
                    if abs(dYaw) >= 1:
                        # print("In else 1")
                        self.leaderComplianceParams['task_haptic_z_torque'] = -1 * np.sign(dYaw)
                    else:
                        # print("In else 2")
                        self.leaderComplianceParams['task_haptic_z_torque'] = -dYaw
            
            # If mobile robot is to left of goal, then haptic torque on leader is applied Counter clockwise
            if curr_close_positions[3] - curr_close_positions[1] > 0:
                self.leaderComplianceParams['task_haptic_z_torque'] = 1
            # If mobile robot is to right of goal, then haptic torque on leader is applied clockwise
            elif curr_close_positions[3] - curr_close_positions[1] < 0:
                self.leaderComplianceParams['task_haptic_z_torque'] = -1
            
            # To save the haptic feedback data
            self.save_mobile_haptic_feedback_x.append(self.leaderComplianceParams['task_haptic_x_force'])
            self.save_mobile_haptic_feedback_yaw.append(self.leaderComplianceParams['task_haptic_z_torque'])

            if self.condition == "h" or self.condition == "vh":
                self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)

    def lidar_conditions(self):
        # The following if conditions checks the lidar messages for obstacles
        if np.mean(self.lidar_ranges) != 0 and np.mean(self.lidar_ranges) <= self.min_obstacle_distance: # m
            self.obstacle = True
            # print("Obstacle present")
            self.obstacle_haptic_force.append(self.leaderComplianceParams['task_haptic_x_force'])
        else:
            self.obstacle = False
            self.obstacle_haptic_force.append(0)

    def hunter_odometry_conditions(self):
        # The following is for hunter odometry data

        if (self.got_hunter_global_plan) and (abs(self.hunter_odom_msg.pose.pose.position.x - self.goal.pose.position.x) < 0.2) and \
                                            (abs(self.hunter_odom_msg.pose.pose.position.y - self.goal.pose.position.y) < 0.2):
            print("Goal Reached in True")
            self.goalReached = True

        if self.got_hunter_global_plan and self.withinWalls:
            self.normal_landing_point(self.hunter_odom_msg.pose, self.hunter_global_path)
        else:
            pass
            # print("Choose the goal position in rviz")

        if not self.initialized_hunter_orientation:
            self.initial_hunter_orientation = self.hunter_odom_msg.pose.pose.orientation

    def leader_motion(self):
        """
        This is the main function for the haptic guided navigation of mobile robot
        """
        self.lidar_conditions()

        # find the current yaw angle of EE on leader side
        q = self.currentLeaderFrankaPose.pose.orientation
        roll, pitch, yaw = tf.transformations.euler_from_quaternion( [q.x, q.y, q.z, q.w ] )
        yawCurrent = yaw / np.pi 

        # find the initial yaw angle of EE on leader side
        q = self.initialLeaderFrankaPose.pose.orientation
        roll, pitch, yaw = tf.transformations.euler_from_quaternion( [q.x, q.y, q.z, q.w ] )
        yawInitial = yaw / np.pi

        # normalized dx (trans) and dz (rot) 
        dx = (  self.currentLeaderFrankaPose.pose.position.x - self.initialLeaderFrankaPose.pose.position.x )
        dz = yawInitial - yawCurrent

        if abs(dx) > self.virtualBoundaryWorkspace:
            print("==recentering please wait=====")
            # self.outsideWorkspace = True
            # self.b_initialLeaderPoseFound = False
            # self.withinWalls = False
        #     # self.initial_position()
            self.homing(5)
            # self.outsideWorkspace = False

        if (not isinstance(self.aruco_msg, type(None))) or (self.goalReached):
            # print("here")
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = 0
            desired_cmd_vel.angular.z = 0
            self.hunterCmdVelPub.publish( desired_cmd_vel )
            print("=========  CONGRATULATIONS - OBJECT DETECTED ================")
            self.end_time = rospy.get_time()
            print("End time: ", self.end_time)
            self.goalReached = True
            
            # self.write_to_file()
            # rospy.sleep(20)

        elif self.obstacle and ((dx > self.virtualBoundaryTrans[0]) and (dx < self.virtualBoundaryTrans[1])):
            # print("Obstclre and within virtual bounds") 
            # When obstacle is present, maximum speed is 0.100 m/s
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = -self.obstacle_speed_scale * (dx / ( self.virtualBoundaryTrans[1] - self.virtualBoundaryTrans[0] ))
            desired_cmd_vel.angular.z = -dz

            if not self.control_follower_franka: self.hunterCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = True
            if not self.time_started:
                self.start_time = rospy.get_time()
                self.time_started = True

        elif dx > self.virtualBoundaryTrans[0] and dx < self.virtualBoundaryTrans[1]:
            # move thorvald forward if (current_franka_x - initial_x_position) > 10 and < 40 cm
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = -self.speed_scale* (dx / ( self.virtualBoundaryTrans[1] - self.virtualBoundaryTrans[0] ))
            desired_cmd_vel.angular.z = -dz

            if not self.control_follower_franka: self.hunterCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = True
            if not self.time_started:
                self.start_time = rospy.get_time()
                self.time_started = True

        elif dx > -1*self.virtualBoundaryTrans[1] and dx < -1 * self.virtualBoundaryTrans[0]: 
            # move thorvald backward if (current_franka_x - initial_x_position) > -40 and < -10 cm
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = - self.speed_scale* (dx / ( self.virtualBoundaryTrans[1] - self.virtualBoundaryTrans[0] ))
            desired_cmd_vel.angular.z = -dz

            if not self.control_follower_franka: self.hunterCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = True
            if not self.time_started:
                self.start_time = rospy.get_time()
                self.time_started = True

        else:
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = 0
            desired_cmd_vel.angular.z = 0
            if not self.control_follower_franka: self.hunterCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = False
            
        self.dz = dz

        self.hunter_odometry_conditions()

    def get_dYaw(self, hunter_orientation, landing_point):
        # find the current yaw angle of EE on leader side
        rollBase, pitchBase, yawBase = tf.transformations.euler_from_quaternion( [hunter_orientation.x, hunter_orientation.y, hunter_orientation.z, hunter_orientation.w ] )
        
        rollPath, pitchPath, yawPath = tf.transformations.euler_from_quaternion( [landing_point.x, landing_point.y, landing_point.z, landing_point.w ] )
        # dYaw = yawPath - yawBase

        # Heading correction: Angle between the current robot orientation and the initial robot orientation 
        rollInit, pitchInit, yawInit = tf.transformations.euler_from_quaternion([ self.initial_hunter_orientation.x, self.initial_hunter_orientation.y, \
            self.initial_hunter_orientation.z, self.initial_hunter_orientation.w])

        # dYaw = np.arctan2(yawPath, yawBase) + np.arctan2(yawInit, yawBase)
        dYaw = yawPath - yawBase
        heading = abs(yawBase) - yawInit
        return dYaw, heading

    def drop_object(self):
        # self.homing(5)
        # self.control_follower_franka = False
        time_to_reach_home = 5.0 # sec
        self.recovery_leader_arm_error()
        self.recovery_follower_arm_error()
        name, state = self.list_controllers("position_joint_trajectory_controller")
        if state == "initialized" or state == "stopped":
            self.switch_controllers("position_joint_trajectory_controller", "panda_leader_cartesian_impedance_controller")
        
        # self.recovery_leader_arm_error()
        # self.recovery_follower_arm_error()


        print("Dropping")

        follower_joint_values = self.follower_curr_joint_states.position
        current_vals = JointTrajectoryPoint()
        current_vals.positions = self.follower_joint_state.q
        # current_vals.positions = [leader_joint_values[0], leader_joint_values[1], leader_joint_values[2],leader_joint_values[3],\
        #  leader_joint_values[4], leader_joint_values[5], leader_joint_values[6]]
        
        # 7 joint vals at home position
        joint_home = JointTrajectoryPoint()
        joint_home.positions = [0.697673392353291, -0.41846365404965585, 0.516888971942968, -1.4136414036010285, 0.2809474829955265, 1.1741948535589524, 0.8438485136611789]
        joint_home.time_from_start = rospy.Time(time_to_reach_home)

        traj = JointTrajectory()
        follower_joint_names = []
        for i in range(7):
            follower_joint_name = "panda_follower"+ "_joint" + str(i+1)
            follower_joint_names.append( follower_joint_name ) 
        traj.joint_names = follower_joint_names
        traj.points = [current_vals, joint_home]

        r = rospy.Rate(1) # 10hz
        self.follower_joint_controller_pub.publish(traj)
        r.sleep()
        self.follower_joint_controller_pub.publish(traj)
        # time.sleep(time_to_reach_home + 0.5) # Code stops for time_to_reach + 0.5 seconds, hence waiting for traj to finish publishing
        # if (leader_joint_values[0] - joint_home.positions[0] < 0.02) and (leader_joint_values[1] - joint_home.positions[1] < 0.02) \
        # and (leader_joint_values[2] - joint_home.positions[2] < 0.02) and (leader_joint_values[3] - joint_home.positions[3] < 0.02) \
        # and (leader_joint_values[4] - joint_home.positions[4] < 0.02) and (leader_joint_values[5] - joint_home.positions[5] < 0.02) \
        # and (leader_joint_values[6] - joint_home.positions[6] < 0.02):
        #     print("Homing Done")
        #     self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")

        if (follower_joint_values[0] - joint_home.positions[0] < 0.06) and (follower_joint_values[1] - joint_home.positions[1] < 0.06) \
        and (follower_joint_values[2] - joint_home.positions[2] < 0.06) and (follower_joint_values[3] - joint_home.positions[3] < 0.06) \
        and (follower_joint_values[4] - joint_home.positions[4] < 0.06) and (follower_joint_values[5] - joint_home.positions[5] < 0.06) \
        and (follower_joint_values[6] - joint_home.positions[6] < 0.06):
            # print("Homing close")
            if self.control_follower_franka == False:
                self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")   
            else:
                rospy.sleep(3)
                # print("Slept for 3 seconds")
                self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
                print("========= YOU CAN NOW CONTROL THE ROBOT=======================")
            print("Dropped object")   
            self.grasp_client()
        else:
            self.drop_object()

    def grasp_client(self):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient(self.follower_ns + '/franka_gripper/grasp', franka_gripper.msg.GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.GraspGoal()
        goal.width = 0.022
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        goal.speed = 0.1
        goal.force = 5

        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()
        self.switch_controllers("position_joint_trajectory_controller", "panda_leader_cartesian_impedance_controller")
        rospy.sleep(20)
    
    def switch_controllers(self, start, stop):
        """
        This function switches controllers. Takes input as strings
        """
        try:
            switch_controller = rospy.ServiceProxy('/panda_leader/controller_manager/switch_controller', SwitchController)
            ret = switch_controller(start_controllers= [start], stop_controllers=[stop], strictness=2, start_asap=False, timeout=0.0)
            # print("=================YOU CAN NOW CONTROL ===============")
        except rospy.ServiceException as e:
            print ("Service call failed: %s",e)
        # if (not self.recorded_init_poses) or (not self.outsideWorkspace):
        #     try:
        #         switch_controller = rospy.ServiceProxy('/panda_leader/controller_manager/switch_controller', SwitchController)
        #         ret = switch_controller(start_controllers= [start], stop_controllers=[stop], strictness=2, start_asap=False, timeout=0.0)
        #         print("=================YOU CAN NOW CONTROL ===============")
        #     except rospy.ServiceException as e:
        #         print ("Service call failed: %s",e)

        # if self.outsideWorkspace:
        #     try:
        #         switch_controller = rospy.ServiceProxy('/panda_leader/controller_manager/switch_controller', SwitchController)
        #         ret = switch_controller(start_controllers= [start], stop_controllers=[stop], strictness=2, start_asap=False, timeout=0.0)
        #         print("=================PLEASE WAIT===============")
        #     except rospy.ServiceException as e:
        #         print ("Service call failed: %s",e)
    
    def list_controllers(self, controller_type):
        try:
            list_controller = rospy.ServiceProxy('/panda_leader/controller_manager/list_controllers', ListControllers)
            ret = list_controller()
            if ret.controller[0].name == controller_type:
                return ret.controller[0].name, ret.controller[0].state
            if ret.controller[1].name == controller_type:
                return ret.controller[1].name, ret.controller[1].state
            if ret.controller[2].name == controller_type:
                return ret.controller[2].name, ret.controller[2].state
        except rospy.ServiceException as e:
            print ("Service call failed: %s",e)

    def write_to_file(self):
        if not self.data_written:
            print("Writing to file")
            # path = os.getcwd()
            # save_path = os.path.join(path, 'test2')
            time_array = [self.start_time, self.end_time]
            time_array_follower = [self.start_follower_time, self.end_follower_time]
            goal = [self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z]

            aruco_goal = self.object_pose()
            follower_goal = [aruco_goal.pose.position.x, aruco_goal.pose.position.y, aruco_goal.pose.position.z, \
            aruco_goal.pose.orientation.x, aruco_goal.pose.orientation.y, aruco_goal.pose.orientation.z, aruco_goal.pose.orientation.w]
            # print("Path is: ", save_path)
            # print("Feedback is : ",self.save_mobile_haptic_feedback_x)
            # print("reference_x is : ",self.save_reference_x)
            # print("reference_y is : ",self.save_reference_y)
            # print("actual_x is : ",self.save_actual_x)
            # print("actual_y is : ",self.save_actual_y)
            np.savez('/home/venkatesh/franka_hunter_ws/src/MobileManip_franka_hunter/franka_hunter_gazebo/script/' + self.user_name + '_' + self.condition + '_' +self.trial_number, \
                reference_x = self.save_reference_x, \
                reference_y = self.save_reference_y, reference_z = self.save_reference_z, actual_x = self.save_actual_x, \
                actual_y = self.save_actual_y, actual_z = self.save_actual_z, mobile_reference_x = self.save_mobile_reference_x, \
                mobile_reference_y = self.save_mobile_reference_y, mobile_actual_x = self.save_mobile_actual_x, mobile_actual_y = self.save_mobile_actual_y, \

                haptic_feedback_x = self.save_haptic_feedback_x, haptic_feedback_y = self.save_haptic_feedback_y, haptic_feedback_z = self.save_haptic_feedback_z,\
                mobile_haptic_feedback_x = self.save_mobile_haptic_feedback_x, mobile_haptic_feedback_yaw = self.save_mobile_haptic_feedback_yaw, \

                obstacle_haptic_force = self.obstacle_haptic_force, external_torques = self.external_torques, follower_external_torque = self.follower_external_torques, \
                
                leader_joint_states = self.save_leader_joints, follower_joint_states = self.save_follower_joints, \

                follower_time = time_array_follower, mobile_time = time_array, mobile_goal = goal, follower_goal = follower_goal)
            # np.savez('/home/venkatesh/haptic_ws/src/thorvald_panda_description/script/strict_landing', self.save_reference_x, self.save_reference_y, self.save_actual_x, self.save_actual_y, self.save_mobile_haptic_feedback_x)
            self.data_written = True
            

    def control_loop(self): 
        
        rospy.loginfo("Press Ctrl+C if anything goes sideways")
        rate = rospy.Rate(100)

        # self.get_times()

        while not rospy.is_shutdown(): 
            try:
                # image = np.zeros((255, 255, 3), np.uint8)
                # image[:0:255] = (0, 0, 255) # Red in BGR format
                image = np.zeros(  ( 200, 200 ) , dtype = np.uint8)
                # image[:,:, 2] = 255
                cv2.imshow( "Switch Control", image ) 
                k = cv2.waitKey(30)
                # esc key
                if k == 27 or k == 1048603:
                    cv2.destroyAllWindows() 
                    break 
                elif k < 0:
                    pass
                else:
                    rospy.loginfo("Key: {}".format(k) )
                    if k == ord(' ') or k == 32:
                        if self.distractor == True:
                            self.distractors_collected += 1
                    if k == ord('q') or k == 113: 
                        rospy.loginfo( "Currently teleoperating the franka arm ")
                        # teleop_manip = checkTeleoperation()
                        # Generate trajectory is the main (root) function while teleoperting the manipulator
                        self.control_follower_franka = True
                        # self.switch_controllers("panda_leader_cartesian_impedance_controller", "position_joint_trajectory_controller")
                    elif k == ord('m'):
                        self.control_follower_franka = False
                    elif k == ord('w') or k == 119:
                        if self.data_written:
                            # teleop_mobile = TeleNavigation()
                            # TODO Check ervything in leader_main function. Currently working on generate_trajectory()
                            
                            # self.all_distractors.append(self.distractors_collected)
                            # print("Visual distractors: ", self.all_distractors)
                            # rospy.loginfo( "Grasped object")
                            self.drop_object()
                        # self.control_follower_franka = False
                    elif k == ord('o') or k == 111:
                        self.goal_reached = True
                        task_end = Bool()
                        task_end.data = True
                        self.pub_end_task.publish(task_end)
                # k = input()
                # if k == "w":
                #     self.end_follower_time = rospy.get_time()
                #     self.goal_reached = True
                # elif k == "o":
                #     self.goalReached = True
                # else:


                # If the aruco marker is detected, control the follower arm as well
                if not isinstance(self.aruco_msg, type(None) ):
                    self.control_follower_franka = True
                    
                    # rospy.loginfo( "Currently teleoperating the franka arm ")     
                # If the aruco marker is not detected, control the mobile robot        
                elif isinstance(self.aruco_msg, type(None) ):
                    self.control_follower_franka = False
                    # rospy.loginfo( "Currently teleoperating the Hunter ")

                if  not isinstance( self.leader_joint_state , type(None) ) and \
                    not isinstance( self.follower_joint_state , type (None) ) :
                    self.franka_leader_main()
                    self.recovery_leader_arm_error()
                    self.recovery_follower_arm_error()
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break

if __name__ == "__main__" : 

    # The below class is useful to perfomr mobile manipulation using franka as leader
    teleop = checkTeleoperation()

    # The below class is useful to perform simple arm-arm mimicking
    # teleop = PandaToPandaTeleoperation()
