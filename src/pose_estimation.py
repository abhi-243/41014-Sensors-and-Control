import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading

# ---------------- Realsense Setup ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

# ---------------- ArUco Setup ----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# ---------------- Camera Intrinsics ----------------
fx, fy = 615, 615
cx, cy = 320, 240
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# ---------------- Board 3D Points (meters) ----------------
object_points = np.array([
    [-0.05, -0.05, 0],
    [0.05, -0.05, 0],
    [0.0, 0.05, 0]
], dtype=np.float32)

# ---------------- Shared data ----------------
latest_board_pos = np.array([0.0, 0.0, 0.0])
latest_ee_pos = np.array([0.0, 0.0, 0.0])
latest_rvec = np.zeros(3)  # EE rotation vector
data_lock = threading.Lock()
running = True

# ---------------- Plotting Thread ----------------
# ---------------- Plotting Thread ----------------
def plot_thread():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    arrow_length = 0.05  # length of EE axes
    while running:
        with data_lock:
            board_pos = latest_board_pos
            ee_pos = latest_ee_pos
            rvec = latest_rvec
        ax.cla()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.5)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # Board position
        ax.scatter(board_pos[0], board_pos[1], board_pos[2], c='red', s=50, label='Board')
        # EE position
        ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], c='blue', s=50, label='UR3 EE')

        # Draw EE axes
        if np.linalg.norm(rvec) > 0:
            R_board, _ = cv2.Rodrigues(rvec)  # Board rotation
            # Assuming EE has same orientation as board
            R_ee = R_board

            # X, Y, Z axes vectors
            axes = np.eye(3) * arrow_length  # unit vectors scaled
            axes_global = R_ee @ axes

            # Plot arrows for EE axes
            ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
                      axes_global[0,0], axes_global[1,0], axes_global[2,0],
                      color='r', linewidth=2, arrow_length_ratio=0.2)
            ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
                      axes_global[0,1], axes_global[1,1], axes_global[2,1],
                      color='g', linewidth=2, arrow_length_ratio=0.2)
            ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
                      axes_global[0,2], axes_global[1,2], axes_global[2,2],
                      color='b', linewidth=2, arrow_length_ratio=0.2)

        ax.legend()
        plt.draw()
        plt.pause(0.001)
    plt.ioff()
    plt.show()

threading.Thread(target=plot_thread, daemon=True).start()

# ---------------- Main Camera Loop ----------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and len(ids) >= 3:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            image_points = []
            for marker_id in [0, 1, 2]:
                if marker_id in ids:
                    idx = np.where(ids == marker_id)[0][0]
                    corner = corners[idx][0]
                    cx, cy = np.mean(corner[:, 0]), np.mean(corner[:, 1])
                    image_points.append([cx, cy])
            image_points = np.array(image_points, dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

            board_pos = tvec.flatten()
            ee_pos = board_pos + np.array([0, 0, 0.2])

            # Update shared data
            with data_lock:
                latest_board_pos = board_pos
                latest_ee_pos = ee_pos
                latest_rvec = rvec

        cv2.imshow('ArUco Detection + Depth', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    running = False
    pipeline.stop()
    cv2.destroyAllWindows()