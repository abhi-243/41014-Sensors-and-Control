import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
import threading
import time

# ---------------- Realsense Setup ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Camera intrinsics (adjust if needed)
fx, fy = 615, 615
cx, cy = 320, 240
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# ---------------- ArUco Setup ----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# ---------------- Board 3D Points (meters) ----------------
object_points = np.array([
    [-0.05, -0.05, 0],
    [0.05, -0.05, 0],
    [0.0, 0.05, 0]
], dtype=np.float32)

# ---------------- UR3 Robot Model ----------------
ur3 = DHRobot([
    RevoluteDH(d=0.1519, a=0, alpha=np.pi/2),
    RevoluteDH(d=0, a=-0.24365, alpha=0),
    RevoluteDH(d=0, a=-0.21325, alpha=0),
    RevoluteDH(d=0.11235, a=0, alpha=np.pi/2),
    RevoluteDH(d=0.08535, a=0, alpha=-np.pi/2),
    RevoluteDH(d=0.0819, a=0, alpha=0)
], name='UR3')

T_desired_rel = SE3(0, 0, 0.2)  # EE desired pose relative to board
dt = 0.05
lam = 1.0
q = ur3.q  # initial joint angles

# ---------------- Matplotlib Setup ----------------
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 0.5)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

running = True
board_pos_shared = np.zeros(3)
board_R_shared = np.eye(3)
data_lock = threading.Lock()

# ---------------- Camera Thread ----------------
def camera_thread():
    global board_pos_shared, board_R_shared, running
    while running:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) >= 3:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            image_points = []
            for marker_id in [0,1,2]:
                if marker_id in ids:
                    idx = np.where(ids==marker_id)[0][0]
                    corner = corners[idx][0]
                    cx_marker, cy_marker = np.mean(corner[:,0]), np.mean(corner[:,1])
                    image_points.append([cx_marker, cy_marker])
            image_points = np.array(image_points, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if success:
                R_board, _ = cv2.Rodrigues(rvec)
                t_board = tvec.flatten()
                with data_lock:
                    board_pos_shared = t_board
                    board_R_shared = R_board
        cv2.imshow('ArUco Detection', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
            break

threading.Thread(target=camera_thread, daemon=True).start()

# ---------------- Main Visual Servoing Loop ----------------
while running:
    with data_lock:
        board_pos = board_pos_shared
        board_R = board_R_shared
    T_board = SE3.Rt(board_R, board_pos)
    T_target = T_board * T_desired_rel
    T_ee = ur3.fkine(q)
    v = lam * (T_ee.inv() * T_target).twist()
    J = ur3.jacobe(q)
    q_dot = np.linalg.pinv(J) @ v
    q = q + q_dot * dt

    # ---------------- Plot ----------------
    ax.cla()
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.scatter(board_pos[0], board_pos[1], board_pos[2], c='red', s=50, label='Board')
    ee_pos = T_ee.t
    ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], c='blue', s=50, label='UR3 EE')
    R = T_ee.R
    arrow_len = 0.05
    axes_global = R @ np.eye(3) * arrow_len
    ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
              axes_global[0,0], axes_global[1,0], axes_global[2,0], color='r', linewidth=2)
    ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
              axes_global[0,1], axes_global[1,1], axes_global[2,1], color='g', linewidth=2)
    ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
              axes_global[0,2], axes_global[1,2], axes_global[2,2], color='b', linewidth=2)
    ax.legend()
    plt.draw()
    plt.pause(dt)

pipeline.stop()
plt.ioff()
plt.show()