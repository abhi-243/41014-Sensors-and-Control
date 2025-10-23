#!/usr/bin/env python3
"""
ur3e_ibvs_node.py

Run this node while:
- ur_driver and controllers are running (joint_group_vel_controller loaded/started)
- realsense2_camera is not required: this node starts its own RealSense pipeline

Replace T_e_cam with your measured end-effector -> camera transform (4x4 numpy)
"""

import rospy
import numpy as np
import cv2
import pyrealsense2 as rs
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from roboticstoolbox import models as rtb_models
from spatialmath import SE3
from IBVS_mod import IBVS_Controller
import time

# ---------- helper: adjoint ----------
def adjoint_from_SE3(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_skew = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[3:, 3:] = R
    adj[3:, :3] = p_skew @ R
    return adj

# ---------- median depth helper ----------
def median_depth_at(depth_frame, cx, cy, radius=1):
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    xs = np.clip(np.arange(int(cx)-radius, int(cx)+radius+1), 0, w-1)
    ys = np.clip(np.arange(int(cy)-radius, int(cy)+radius+1), 0, h-1)
    vals = []
    for x in xs:
        for y in ys:
            d = depth_frame.get_distance(x, y)
            if d and d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0

class UR3EIBVSNode:
    def __init__(self):
        rospy.init_node("ur3e_ibvs_follower", anonymous=True)

        # params (tune these)
        self.rate_hz = rospy.get_param("~rate_hz", 25.0)
        self.lambda_ = rospy.get_param("~lambda", 0.05)
        self.num_markers = rospy.get_param("~num_markers", 3)
        self.max_joint_speed = rospy.get_param("~max_joint_speed", 0.2)  # rad/s
        self.smooth_alpha = rospy.get_param("~smooth_alpha", 0.6)
        self.depth_patch = rospy.get_param("~depth_patch", 1)
        self.lost_frames_threshold = rospy.get_param("~lost_frames_threshold", 2)

        # robot model & state
        self.robot = rtb_models.UR3e()
        self.q = np.zeros(6)
        rospy.Subscriber('/joint_states', JointState, self.joint_cb, queue_size=1)

        # publisher to joint_group_vel_controller
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=1)

        # IBVS controller
        # pass a placeholder Jacobian; will update each loop
        profile = self._rs_profile()
        self.ibvs = IBVS_Controller(robot_jacobian=np.eye(6), proportional_constant=self.lambda_, camera_profile=profile, featureCount=self.num_markers)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # hand-eye transform T_e_cam: replace identity with measured 4x4 from CAD
        self.T_e_cam = np.eye(4)  # << REPLACE with true transform in meters

        # state
        self.target_set = False
        self.prev_qdot = np.zeros(6)
        self.lost_count = 0

        rospy.loginfo("UR3E IBVS node initialized")

    def _rs_profile(self):
        # quick helper to get profile for IBVS init
        tmp_pipe = rs.pipeline()
        tmp_cfg = rs.config()
        tmp_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        p = tmp_pipe.start(tmp_cfg)
        tmp_pipe.stop()
        return p

    def joint_cb(self, msg):
        self.q = np.array(msg.position[:6])

    def clamp_norm(self, arr, max_norm):
        mag = np.linalg.norm(arr)
        if mag <= max_norm:
            return arr
        return arr * (max_norm / mag)

    def loop(self):
        rate = rospy.Rate(self.rate_hz)
        cv2.namedWindow("UR3E IBVS")

        try:
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    rate.sleep(); continue

                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                if ids is None or len(corners) < self.num_markers:
                    # not enough markers visible
                    self.lost_count += 1
                    if self.lost_count > self.lost_frames_threshold:
                        # publish zero velocities and reset target
                        self.vel_pub.publish(Float64MultiArray(data=[0.0]*6))
                        self.target_set = False
                    cv2.putText(color_image, "Markers lost", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow("UR3E IBVS", color_image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    rate.sleep()
                    continue
                else:
                    self.lost_count = 0

                # sort by marker ID to ensure consistent ordering
                ids_flat = ids.flatten()
                sort_idx = np.argsort(ids_flat)
                corners_sorted = [corners[i] for i in sort_idx]
                ids_sorted = ids_flat[sort_idx]

                centroids = []
                depths = []
                for c in corners_sorted[:self.num_markers]:
                    pts = c[0]
                    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
                    z = median_depth_at(depth_frame, cx, cy, radius=self.depth_patch)
                    if z == 0.0:
                        # fallback: skip control this frame
                        z = median_depth_at(depth_frame, cx, cy, radius=self.depth_patch+1)
                    centroids.append([cx, cy])
                    depths.append(z)
                    cv2.circle(color_image, (int(cx), int(cy)), 3, (0,255,0), -1)

                # calibration: press 'c' to set target centroids (ordered)
                cv2.putText(color_image, "Press 'c' to set target", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.imshow("UR3E IBVS", color_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and not self.target_set:
                    self.ibvs.set_target_features(centroids)
                    self.target_set = True
                    rospy.loginfo("Target calibrated from visible markers (ordered by ID).")
                    continue
                elif key == 27:
                    break

                if not self.target_set:
                    rate.sleep(); continue

                # compute image jacobians with per-feature depths (meters)
                try:
                    imageJs = self.ibvs.compute_image_jacobians(centroids, Zs=depths)
                    v_cam = self.ibvs.calculate_camera_velocities(imageJs, np.array(centroids))  # 6x1
                except Exception as e:
                    rospy.logwarn("IBVS compute error: {}".format(e))
                    rate.sleep(); continue

                # only keep lateral translation in camera frame (x,y). zero z and all rotations
                v_cam = v_cam.reshape(6)
                v_cam[2] = 0.0
                v_cam[3:] = 0.0
                v_cam = v_cam.reshape(6,1)

                # transform camera twist to base frame: twist_base = Ad(T_b_c) * v_cam
                T_b_e = self.robot.fkine(self.q).A
                T_b_c = T_b_e @ self.T_e_cam
                Ad_b_c = adjoint_from_SE3(T_b_c)
                twist_base = Ad_b_c @ v_cam  # 6x1 in base frame

                # compute qdot via pseudoinverse of jacobe (robot.jacobe maps q_dot -> twist_base)
                J = self.robot.jacobe(self.q)
                qdot = np.linalg.pinv(J) @ twist_base
                qdot = qdot.flatten()

                # smoothing (exponential) and clamp
                qdot_sm = self.smooth_alpha * self.prev_qdot + (1 - self.smooth_alpha) * qdot
                qdot_sm = self.clamp_norm(qdot_sm, self.max_joint_speed)
                self.prev_qdot = qdot_sm.copy()

                # publish
                msg = Float64MultiArray()
                msg.data = qdot_sm.tolist()
                self.vel_pub.publish(msg)

                rate.sleep()

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            rospy.loginfo("UR3E IBVS node terminated")

if __name__ == "__main__":
    node = UR3EIBVSNode()
    node.loop()