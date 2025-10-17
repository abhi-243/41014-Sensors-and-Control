import pyrealsense2 as rs
import numpy as np
import cv2

# ---- Realsense setup ----
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)   # Align depth to color frame

# ---- ArUco setup ----
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

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

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i, corner in enumerate(corners):
                pts = corner[0]
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                depth = depth_frame.get_distance(int(cx), int(cy))

                # Draw markers + display distance
                cv2.circle(color_image, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(color_image, f"ID {int(ids[i])}: {depth:.3f} m",
                            (int(cx) - 40, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('ArUco Detection + Depth', color_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()