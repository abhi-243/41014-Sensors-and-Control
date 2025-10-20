#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from roboticstoolbox import ERobot, tools as rtb_tools
from spatialmath import SE3

def rmrc(robot, q_start, target_pose, dt=0.05, Kp=1.0, max_iter=200):
    """
    RMRC using Robotics Toolbox
    :param robot: ERobot instance
    :param q_start: initial joint angles
    :param target_pose: SE3 object (desired EE pose)
    :param dt: integration time step
    :param Kp: proportional gain
    :param max_iter: max iterations
    :return: list of joint positions
    """
    q = np.array(q_start)
    trajectory = []

    for _ in range(max_iter):
        # Current EE pose
        T = robot.fkine(q)
        
        # Position error
        pos_err = target_pose.t - T.t

        # Orientation error (rotation vector)
        R_err = target_pose.R @ T.R.T
        orient_err = rtb_tools.tr2rpy(R_err)  # or use R_err.log() for rotvec

        # Full 6D error
        e = np.hstack([pos_err, orient_err])
        
        if np.linalg.norm(e) < 1e-3:
            break
        
        # Jacobian
        J = robot.jacobe(q)  # 6x6 Jacobian (EE in base frame)
        
        # Compute joint velocities
        dq = Kp * np.linalg.pinv(J) @ e
        
        # Integrate
        q = q + dq * dt
        trajectory.append(q.copy())
    
    return trajectory

def main():
    rospy.init_node('rmrc_trajectory')

    # Connect to trajectory controller
    client = actionlib.SimpleActionClient(
        '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    rospy.loginfo("Waiting for action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to controller.")

    # Create UR robot model from Robotics Toolbox
    robot = ERobot.ur3e()

    # Initial joint configuration
    q_start = [0, -np.pi/2, np.pi/2, -np.pi/2, 0, 0]

    # Desired end-effector pose as SE3 object
    target_pose = SE3(0.4, 0.2, 0.3) * SE3.OA([1,0,0], [0,1,0])  # position + orientation

    # Compute RMRC trajectory
    traj = rmrc(robot, q_start, target_pose)

    # Build ROS trajectory goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    for i, q in enumerate(traj):
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        point.time_from_start = rospy.Duration(dt * (i + 1))
        goal.trajectory.points.append(point)

    client.send_goal(goal)
    rospy.loginfo("RMRC trajectory sent")
    client.wait_for_result()
    rospy.loginfo("Execution complete")

if __name__ == '__main__':
    main()