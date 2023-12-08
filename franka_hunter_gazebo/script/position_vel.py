#!/usr/bin/env python

import sys
import time 
import rospy
import tf.transformations
import numpy as np
from math import atan2, exp
import os

from geometry_msgs.msg import Pose, PoseStamped, Twist, PoseWithCovarianceStamped
from franka_msgs.msg import FrankaState
from franka_control.msg import ErrorRecoveryActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control_msgs.msg import FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal

from visualization_msgs.msg import Marker
from nav_msgs.msg import Path , Odometry

from sensor_msgs.msg import LaserScan

import moveit_commander
import moveit_msgs.msg

# dynamic reconfigure to set stiffness and damping
import dynamic_reconfigure.client

# Switch between position and impedance control
from controller_manager_msgs.srv import SwitchController

class TeleNavigation():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("telenaviagation_arm_to_thorvald_node")

        # Variable declaration
        self.b_initialLeaderPoseFound = False
        self.initialLeaderFrankaPose = PoseStamped()
        self.currentLeaderFrankaPose = PoseStamped()
        self.leaderComplianceParams = None
        
        self.withinWalls = False
        self.globalPath = Path()

        self.goal = PoseStamped()
        self.goalReached = False
        self.got_thorvald_goal = False
        self.thorvald_global_path = Path()
        self.got_thorvald_global_plan = False
        self.initial_position_estimated = False
        self.initialized_thorvald_orientation = False
        self.initial_thorvald_orientation = 0

        self.linkName = "panda_link0"
        self.virtualBoundaryTrans = [0.1, 0.4] # m
        self.virtualBoundaryRot = [0.1, 0.4] # m
        self.virtualTranslationalStiffnessLimits = [0, 100] # N-m
        self.virtualRotationalStiffnessLimtis = [0, 30] # N-rad
        # Virtual boundary limiting workspace along x
        self.virtualBoundaryWorkspace = 0.2 # m
        self.outsideWorkspace = False

        # Feedback limit params
        # Checking force applied by user on EE
        self.dz = 0.0
        self.high_external_torque = False
        self.external_torques = []

        self.positionLimitsForInitialPose = [[-0.6, 0.6], [-0.6, 0.6], [0.2, 0.9]]

        self.too_close = False

        self.at_initial_position = False

        # Obstacle detection params intialization
        self.obstacle = False
        self.min_obstacle_distance = 1.5 # m
        self.lidar_ranges = 0 # m
        self.obstacle_haptic_force = []

        # Landing point params 
        self.threshold = 0.5
        self.have_merge_point = False

        # Saving data
        self.save_haptic_feedback_x = []
        self.save_haptic_feedback_yaw = []
        self.save_actual_x = []
        self.save_actual_y = []
        self.save_reference_x = []
        self.save_reference_y = []

        # Time
        self.time_started = False
        self.start_time = 0
        self.end_time = 0


        # Moveit initialization
        self.leader_commander = moveit_commander.RobotCommander()
        self.leader_group = moveit_commander.MoveGroupCommander("panda_arm")

        # Publishers
        self.setPosePub = rospy.Publisher("/leader/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        self.leaderFrankaRecoveryPub = rospy.Publisher('/leader/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10)
        self.goalPub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # self.leaderStartPositionPub = rospy.Publisher('/leader/position_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
        
        self.followerCmdVelPub = rospy.Publisher( "/keyboard_joy/cmd_vel", Twist, queue_size=10)
        self.desiredPathPub = rospy.Publisher("/global_path", Path, queue_size = 1)
        self.markerVisualisePub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        # Initial pose publisher
        self.initialThorvaldPosePub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)

        # # Subscribers
        self.leaderFrankaStateSub = rospy.Subscriber("/leader/franka_state_controller/franka_states", FrankaState, self.leader_franka_state_cb)
        # self.thorvaldNavGoalSub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.thorvald_nav_goal_cb)
        self.thorvaldVisualPathSub = rospy.Subscriber("/move_base/DWAPlannerROS/global_plan", Path, self.thorvald_nav_path_cb)
        self.leaderThovaldOdomSub = rospy.Subscriber("/odometry/base_raw", Odometry, self.follower_thorvald_odom_cb)
        self.lidarSub = rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.leaderComplinaceConfigServer = dynamic_reconfigure.client.Client("/leader/leader_cartesian_impedance_controller_controller_params", timeout=30, config_callback=self.compliance_config_cb)

        self.handle_loop()


    def compliance_config_cb(self, config):
        self.leaderComplianceParams = config

    def thorvald_nav_goal_cb(self, msg):
        """
        This function gets the pose of the 2d navigation point set by the user
        """
        if isinstance(msg.pose.position.x, float):
            self.goal = msg
            self.got_thorvald_goal = True

    def thorvald_nav_path_cb(self, msg):
        """
        This function gets the path to goal generated by the global planner
        """
        if (self.got_thorvald_goal) and (not self.got_thorvald_global_plan):
            print("Got thorvald plan")            
            self.got_thorvald_global_plan = True
            self.initialized_thorvald_orientation = True

        self.thorvald_global_path = msg


    def leader_franka_state_cb(self, msg):
        """
        This function defines the intial pose the franka robot is present in, as the reference.
        Next, the haptic feedback functions are called
        """
        current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
        current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)

        pose = PoseStamped()
        pose.header.frame_id = self.linkName
        pose.header.stamp = rospy.Time(0)
        pose.pose.orientation.x = current_quaternion[0]
        pose.pose.orientation.y = current_quaternion[1]
        pose.pose.orientation.z = current_quaternion[2]
        pose.pose.orientation.w = current_quaternion[3]
        pose.pose.position.x = msg.O_T_EE[12]
        pose.pose.position.y = msg.O_T_EE[13]
        pose.pose.position.z = msg.O_T_EE[14]

        external_torque = msg.tau_ext_hat_filtered

        # Record torques at EE when the goal is defined but not reached
        if (self.got_thorvald_goal) and (not self.goalReached):
            self.external_torques.append(external_torque[6])

        # Check for torque due to external forces on EE in Nm
        if abs(external_torque[6]) >= 1.0:
            # print(" ")
            # print("tau_ext_hat_filtered", external_force)
            self.high_external_torque = True
        else:
            self.high_external_torque = False

        if not (self.b_initialLeaderPoseFound):
            self.initial_position()
            if not self.initial_position_estimated:
                self.initial_thorvald_position()
                self.initial_position_estimated = True
            self.initialLeaderFrankaPose = pose
            self.b_initialLeaderPoseFound = True
        else: 
            self.currentLeaderFrankaPose = pose
            self.leader_motion() # haptic navigation


    def initial_thorvald_position(self):

        thorvald_initpose = PoseWithCovarianceStamped()
        thorvald_initpose.header.frame_id = "map"
        thorvald_initpose.pose.pose.position.x = -2.1709985733
        thorvald_initpose.pose.pose.position.y = -0.539679527283
        thorvald_initpose.pose.pose.position.z = 0.0
        thorvald_initpose.pose.pose.orientation.x = 0.0
        thorvald_initpose.pose.pose.orientation.y = 0.0
        thorvald_initpose.pose.pose.orientation.z = 0.0
        thorvald_initpose.pose.pose.orientation.w = 1.0
        self.initialThorvaldPosePub.publish(thorvald_initpose)

        desired_cmd_vel = Twist()
        desired_cmd_vel.linear.x = 0
        desired_cmd_vel.angular.z = 0
        self.followerCmdVelPub.publish( desired_cmd_vel )

    def initial_position(self):
        """
        This function moves the robot to the center (initial position) in the beginning and when wrokspace limits are exceeded
        """

        # recover leader franka 
        error_msg = ErrorRecoveryActionGoal()
        self.leaderFrankaRecoveryPub.publish( error_msg ) 
        # Moveit code
        joint_goal = self.leader_group.get_current_joint_values()
        # print(self.leader_commander.get_current_state())
        joint_goal[0] = -0.12
        joint_goal[1] = -0.45
        joint_goal[2] = -0.06
        joint_goal[3] = -2.39
        joint_goal[4] = -0.02
        joint_goal[5] = 1.96
        joint_goal[6] = 0.85
        self.leader_group.go(joint_goal, wait=True)


        if abs(self.leader_commander.get_current_state().joint_state.position[1] + 0.45) < 0.04:
            rospy.sleep(2)
            self.outsideWorkspace = False
            print(" Reached Initial position")
            self.switch_controllers()
        else:
            self.initial_position()
        


    def leader_motion(self):
        """
        This function takes in the joint states of franka robot as input and converts them to velocities
        If Franka is moved more than 10cm in the x direction, then thorvald moves
        Next, the global path plan function is called
        """
        current_trans_stiffness = self.leaderComplianceParams['translational_x_stiffness']
        current_rot_stiffness = self.leaderComplianceParams['rotational_z_stiffness']

        # basically compliance
        translation_factor = 1 - current_trans_stiffness / (self.virtualTranslationalStiffnessLimits[1] - self.virtualTranslationalStiffnessLimits[0])
        rotational_factor = 1 - current_rot_stiffness / (self.virtualRotationalStiffnessLimtis[1] - self.virtualRotationalStiffnessLimtis[0])

        # recover leader franka 
        error_msg = ErrorRecoveryActionGoal()
        self.leaderFrankaRecoveryPub.publish( error_msg ) 

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
            self.outsideWorkspace = True
            self.switch_controllers()
            self.b_initialLeaderPoseFound = False
            self.withinWalls = False
            # self.initial_position()
            # self.outsideWorkspace = False

        if self.goalReached:
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = 0
            desired_cmd_vel.angular.z = 0
            self.followerCmdVelPub.publish( desired_cmd_vel )
            print("=========  CONGRATULATIONS - GOAL REACHED ================")
            self.end_time = rospy.get_time()
            self.write_to_file()
            rospy.sleep(10)

        elif dx > self.virtualBoundaryTrans[0] and dx < self.virtualBoundaryTrans[1]: 
            # move thorvald forward if (current_franka_x - initial_x_position) > 10 and < 40 cm
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = -dx / ( self.virtualBoundaryTrans[1] - self.virtualBoundaryTrans[0] )
            desired_cmd_vel.angular.z = -dz

            self.followerCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = True
            if not self.time_started:
                self.start_time = rospy.get_time()
                self.time_started = True


        elif dx > -1*self.virtualBoundaryTrans[1] and dx < -1 * self.virtualBoundaryTrans[0]: 
            # move thorvald backward if (current_franka_x - initial_x_position) > -40 and < -10 cm
            desired_cmd_vel = Twist()
            desired_cmd_vel.linear.x = -dx / ( self.virtualBoundaryTrans[1] - self.virtualBoundaryTrans[0] )
            desired_cmd_vel.angular.z = -dz

            self.followerCmdVelPub.publish( desired_cmd_vel )
            self.withinWalls = True
            if not self.time_started:
                self.start_time = rospy.get_time()
                self.time_started = True
        else: 
            self.withinWalls = False

        self.dz = dz
        # print("dz: ",self.dz)

        # Visualize goal position
        if self.got_thorvald_goal:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.position.x = self.goal.pose.position.x
            marker.pose.position.y = self.goal.pose.position.y
            marker.pose.position.z = self.goal.pose.position.z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            self.markerVisualisePub.publish(marker)
            # self.goal_visualized = True

        # self.create_global_path_plan(_current, _goal)

    def switch_controllers(self):
        if (not self.b_initialLeaderPoseFound) or (not self.outsideWorkspace):
            try:
                switch_controller = rospy.ServiceProxy('/leader/controller_manager/switch_controller', SwitchController)
                ret = switch_controller(['leader_cartesian_impedance_controller'], ['position_joint_trajectory_controller'], 2)
                print("=================YOU CAN NOW CONTROL ===============")
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        if self.outsideWorkspace:
            try:
                switch_controller = rospy.ServiceProxy('/leader/controller_manager/switch_controller', SwitchController)
                ret = switch_controller(['position_joint_trajectory_controller'], ['leader_cartesian_impedance_controller'], 2)
                print("=================PLEASE WAIT===============")
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

    def create_global_path_plan(self, current_state, goal_state):
        """
        This function creates the path for the thorvald robot to follow
        """
        numPoints = 100 # 100, so that the goal position will be in meters

        x_current, y_current, z_current = current_state.pose.position.x, current_state.pose.position.y, current_state.pose.position.z
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

        self.desiredPathPub.publish( path ) 
        self.globalPath = path

    def lidar_cb(self, msg):
        # This function takes in lidar values as input and provides haptic_feedback when obstacle is very close
        self.lidar_ranges = msg.ranges
        if np.mean(self.lidar_ranges) != 0 and np.mean(self.lidar_ranges) <= self.min_obstacle_distance: # m
            self.obstacle = True
            self.obstacle_haptic_force.append(self.leaderComplianceParams['task_haptic_x_force'])
        else:
            self.obstacle = False
            self.obstacle_haptic_force.append(0)


    def follower_thorvald_odom_cb(self, msg):
        """
        This function adjusts the stiffness of the franka robot based on the current position of Franka
        TODO: Check if the value of 0.2 is fine. Initially was 0.03
        """
        # offset_x = abs(msg.pose.pose.position.x - self.goal.pose.position.x) - 2.1709985733
        # offset_y = abs(msg.pose.pose.position.y - self.goal.pose.position.y) - 0.539679527283

        offset_x = msg.pose.pose.position.x - (-2.1709985733)
        offset_y = msg.pose.pose.position.y - (-0.539679527283)


        if (self.got_thorvald_global_plan) and (abs(msg.pose.pose.position.x - self.goal.pose.position.x) < 0.2) and (abs(msg.pose.pose.position.y - self.goal.pose.position.y) < 0.2):
            print("Goal Reached in True")
            self.goalReached = True

        if self.got_thorvald_global_plan and self.withinWalls:
            # self.find_landing_point(msg.pose, self.thorvald_global_path)
            self.normal_landing_point(msg.pose, self.thorvald_global_path)
        else:
            pass
            # print("Choose the goal position in rviz")

        if not self.initialized_thorvald_orientation:
            self.initial_thorvald_orientation = msg.pose.pose.orientation

        # if self.withinWalls:
        #     self.haptic_feedback(position_vals, dYaw)
        # else: 
        #     self.leaderComplianceParams['task_haptic_z_torque'] = 0
        #     self.leaderComplianceParams['rotational_z_stiffness'] = 0
        #     self.leaderComplianceParams['translational_x_stiffness'] = 0
        #     self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)


    def normal_landing_point(self, point, plan):
        """
        Landing point is the closest point on the planned trajectory in front of the robot's current position that can be reached smoothly
        """
        minEulerDistance = np.Inf
        position_vals = []
        landing_point_viz = []
        for i in range( len(plan.poses) ):
            x1, y1 = point.pose.position.x , point.pose.position.y  # Thorvald's base coordinates
            x2, y2 = plan.poses[i].pose.position.x, plan.poses[i].pose.position.y   # The trajectory's coordinates      
                
            # Check all points on the generated path that are ahead of the robot's base
            if x2 > x1:
                # print("Goal ahead")
                dist = np.sqrt( np.square(x2 - x1) + np.square(y2 - y1) )
                if dist < minEulerDistance:
                    minEulerDistance = dist
                    
                    # Get dYaw based on the landing point and the thorvald orientation
                    try:
                        thorvald_orientation = point.pose.orientation
                        landing_point = plan.poses[i+40].pose.orientation
                        landing_point_pos = plan.poses[i+40].pose.position
                        dYaw, heading = self.get_dYaw(thorvald_orientation, landing_point)
                    except IndexError:
                        # The goal is in front of the robot but closer than 40 steps ahead
                        self.too_close = True
                        thorvald_orientation = point.pose.orientation
                        landing_point = plan.poses[i].pose.orientation
                        landing_point_pos = plan.poses[i].pose.position
                        dYaw, heading = self.get_dYaw(thorvald_orientation, landing_point)

                    landing_point_viz.append(landing_point_pos)
                    position_vals.append([x1, y1, landing_point_pos.x, landing_point_pos.y])
            # If the robot is front of goal then the landing point is the goal
            else:
                thorvald_orientation = point.pose.orientation
                landing_point = self.goal.pose.orientation
                landing_point_pos = self.goal.pose.position
                dYaw, heading = self.get_dYaw(thorvald_orientation, landing_point)
                # Landing point x, y are the goal's x and y
                x2 = self.goal.pose.position.x
                y2 = self.goal.pose.position.y

                landing_point_viz.append(landing_point_pos)
                position_vals.append([x1, y1, x2, y2])

        landing_point_viz = landing_point_viz[-1]
        position_vals = position_vals[-1]
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = position_vals[2]
        marker.pose.position.y = position_vals[3]
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.markerVisualisePub.publish(marker)
        self.save_reference_x.append(position_vals[2])
        self.save_reference_y.append(position_vals[3])
        self.save_actual_x.append(position_vals[0])
        self.save_actual_y.append(position_vals[1])
        self.haptic_feedback(position_vals, dYaw, heading)


    def get_dYaw(self, thorvald_orientation, landing_point):
        # find the current yaw angle of EE on leader side
        rollBase, pitchBase, yawBase = tf.transformations.euler_from_quaternion( [thorvald_orientation.x, thorvald_orientation.y, thorvald_orientation.z, thorvald_orientation.w ] )
        
        rollPath, pitchPath, yawPath = tf.transformations.euler_from_quaternion( [landing_point.x, landing_point.y, landing_point.z, landing_point.w ] )
        # dYaw = yawPath - yawBase

        # Heading correction: Angle between the current robot orientation and the initial robot orientation 
        rollInit, pitchInit, yawInit = tf.transformations.euler_from_quaternion([ self.initial_thorvald_orientation.x, self.initial_thorvald_orientation.y, \
            self.initial_thorvald_orientation.z, self.initial_thorvald_orientation.w])

        # dYaw = np.arctan2(yawPath, yawBase) + np.arctan2(yawInit, yawBase)
        dYaw = yawPath - yawBase
        heading = abs(yawBase) - yawInit
        return dYaw, heading


    def haptic_feedback(self, curr_close_positions, dYaw, heading):
        """
        This functions takes into account the path to the goal position and gives feedback on the leader arm
        """
        if ((self.goalReached) and (self.got_thorvald_global_plan)) or (self.too_close):
            self.leaderComplianceParams['task_haptic_x_force'] = 0
            self.leaderComplianceParams['task_haptic_z_torque'] = 0
            self.too_close = False

        # Haptic force pushing backward when obstacle is in front
        elif self.obstacle:
            self.leaderComplianceParams['task_haptic_x_force'] = 5 * np.mean(self.lidar_ranges)  
            self.leaderComplianceParams['task_haptic_z_torque'] = 0

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
            # i.e task_haptic_z_torque is +ve (EE turns CCW i.e towards franka C)
            
            # else:
            # If an external torque of more than 1Nm is applied, then the feedback decreases exponentially
            if self.high_external_torque:
                print("High external")
                self.leaderComplianceParams['task_haptic_z_torque'] = 1 / (3*exp(abs(dYaw)))
            elif abs(self.dz) >= 0.5:
                self.leaderComplianceParams['task_haptic_z_torque'] = 0
            else:
                if abs(dYaw) >= 1:
                    self.leaderComplianceParams['task_haptic_z_torque'] = -1 * np.sign(dYaw)
                else:
                    self.leaderComplianceParams['task_haptic_z_torque'] = -dYaw
        
        # If mobile robot is to left of goal, then haptic torque on leader is applied Counter clockwise
        # if curr_close_positions[3] - curr_close_positions[1] > 0:
            # self.leaderComplianceParams['task_haptic_z_torque'] = 1
        # If mobile robot is to right of goal, then haptic torque on leader is applied clockwise
        # elif curr_close_positions[3] - curr_close_positions[1] < 0:
            # self.leaderComplianceParams['task_haptic_z_torque'] = -1
        
        # To save the haptic feedback data
        self.save_haptic_feedback_x.append(self.leaderComplianceParams['task_haptic_x_force'])
        self.save_haptic_feedback_yaw.append(self.leaderComplianceParams['task_haptic_z_torque'])

        self.leaderComplinaceConfigServer.update_configuration(self.leaderComplianceParams)

    def goal_position(self, goal_pos):
        self.goal = PoseStamped()

        # Goal close
        if goal_pos == 1:
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = 3.1922
            self.goal.pose.position.y = -2.86616
            self.goal.pose.position.z = 0.0
            self.goal.pose.orientation.x = 0
            self.goal.pose.orientation.y = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1
        # Goal is far behind small obstacle
        elif goal_pos == 2:
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = 7.80711
            self.goal.pose.position.y = -7.36641
            self.goal.pose.position.z = 0.0
            self.goal.pose.orientation.x = 0
            self.goal.pose.orientation.y = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1
        # Goal is to the right of two obstacles
        elif goal_pos == 3:
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = 2.66916
            self.goal.pose.position.y = -8.25532
            self.goal.pose.position.z = 0.0
            self.goal.pose.orientation.x = 0
            self.goal.pose.orientation.y = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1
        # Goal is behind wall
        elif goal_pos == 4:
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = 5.31661
            self.goal.pose.position.y = 0.14718
            self.goal.pose.position.z = 0.0
            self.goal.pose.orientation.x = 0
            self.goal.pose.orientation.y = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1
        # Goal is behind patch
        elif goal_pos == 5:
            self.goal.header.frame_id = "map"
            self.goal.pose.position.x = 7.70116
            self.goal.pose.position.y = 5.257309
            self.goal.pose.position.z = 0.0
            self.goal.pose.orientation.x = 0
            self.goal.pose.orientation.y = 0
            self.goal.pose.orientation.z = 0
            self.goal.pose.orientation.w = 1

        self.goalPub.publish(self.goal)
        self.got_thorvald_goal = True


    def write_to_file(self):
        # path = os.getcwd()
        # save_path = os.path.join(path, 'test2')
        time_array = [self.start_time, self.end_time]
        goal = [self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z]
        # print("Path is: ", save_path)
        # print("Feedback is : ",self.save_haptic_feedback_x)
        # print("reference_x is : ",self.save_reference_x)
        # print("reference_y is : ",self.save_reference_y)
        # print("actual_x is : ",self.save_actual_x)
        # print("actual_y is : ",self.save_actual_y)
        np.savez('/home/venkatesh/haptic_ws/src/thorvald_panda_description/script/i_40_obstacle_overshoot', reference_x = self.save_reference_x, \
            reference_y = self.save_reference_y, actual_x = self.save_actual_x, actual_y = self.save_actual_y, haptic_feedback_x = self.save_haptic_feedback_x, \
            haptic_feedback_yaw = self.save_haptic_feedback_yaw, obstacle_haptic_force = self.obstacle_haptic_force, external_torques = self.external_torques, time = time_array, goal = goal)
        # np.savez('/home/venkatesh/haptic_ws/src/thorvald_panda_description/script/strict_landing', self.save_reference_x, self.save_reference_y, self.save_actual_x, self.save_actual_y, self.save_haptic_feedback_x)

    def handle_loop(self):
        self.goal_position(1)
        while True:
            try:
                # while not self.at_initial_position:
                #     self.initial_position()
                rospy.sleep(1)

            except KeyboardInterrupt:
                self.leaderFrankaStateSub.unregister()
                self.leaderThovaldOdomSub.unregister()
                self.thorvaldNavGoalSub.unregister()
                self.thorvaldVisualPathSub.unregister()
                self.lidarSub.unregister()
                break

def talker():
    
    print("In talker")
    rospy.init_node('move_robot', anonymous = True)
    leaderStartPositionPub = rospy.Publisher('/leader/franka_state_controller/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(10)

    states = JointState()
    states.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
    states.position = [-0.12, -0.45, -0.06, -2.39, -0.02, 1.96, 0.85]

    while not rospy.is_shutdown():
        print("In while loop")
        leaderStartPositionPub.publish(states)
        rate.sleep()


if __name__== '__main__':
    # talker()
    thorvald_navigation = TeleNavigation()