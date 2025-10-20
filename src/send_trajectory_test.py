#!/usr/bin/env python3
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

def main():
    rospy.init_node('send_trajectory')

    # Create an action client to communicate with the controller
    client = actionlib.SimpleActionClient(
        '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )

    rospy.loginfo("Waiting for action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to trajectory controller.")

    # Create a goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    # Define a single waypoint
    point = JointTrajectoryPoint()
    point.positions = [0.0, -1.57, 1.57, -1.57, 0.0, 0.0]  # example pose
    point.time_from_start = rospy.Duration(3.0)
    goal.trajectory.points.append(point)

    # Send the goal
    client.send_goal(goal)
    rospy.loginfo("Trajectory sent, waiting for result...")
    client.wait_for_result()
    rospy.loginfo("Trajectory execution complete.")

if __name__ == '__main__':
    main()