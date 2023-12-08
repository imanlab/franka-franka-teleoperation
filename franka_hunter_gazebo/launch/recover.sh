#!/bin/bash
echo 'rostopic pub --once /leader/franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"'
