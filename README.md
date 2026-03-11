# 41014 Sensors and Control — UR3 Eye-in-Hand Visual Servoing

**Subject:** 41014 Sensors and Control for Mechatronic Systems  
**University:** University of Technology Sydney (UTS)  
**Author:** Abhi Naglapura, Hamish Judson, Dylan Bitar, Jacob Bakhos
**License:** MIT

---

## Overview

This project implements an **image-based visual servoing (IBVS)** system on a Universal Robots UR3 collaborative robot arm. An RGB-D camera is rigidly mounted on the robot's end-effector (eye-in-hand configuration), and real-time image feedback is used to guide the robot's motion toward a visual target — an **ArUco marker**.

The system continuously detects ArUco markers in the camera's field of view, computes the error between the current and desired image features, and generates corrective velocity commands to drive the robot until the target is centred at the correct distance.

Demo footage is included in `camera_detection.mp4` and `1000010793.mp4`.

---

## Repository Structure

```
41014-Sensors-and-Control/
│
├── src/                        # Main Python source code
├── .vscode/                    # VSCode workspace settings
│
├── CMakeLists.txt              # ROS catkin build configuration
├── package.xml                 # ROS package manifest
├── requirements.txt            # Python dependencies
├── ur3_spinup                  # Script to initialise the UR3 connection
│
├── marker_0.png                # ArUco marker ID 0 (print and use as target)
├── marker_1.png                # ArUco marker ID 1
├── marker_2.png                # ArUco marker ID 2
├── a4_markers_square.png       # A4 printable ArUco marker sheet
│
├── camera_detection.mp4        # Demo: marker detection in action
├── 1000010793.mp4              # Demo: full visual servoing behaviour on UR3
│
├── HOW TO RUN PYTHON CODE ON THE UR3 - Docker.txt   # Docker setup guide
└── LICENSE
```

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Robot Arm | Universal Robots UR3 (6-DOF) |
| Camera | RGB-D camera (e.g. Intel RealSense D435/D435i) |
| Mount | Rigid end-effector camera mount (eye-in-hand) |
| Visual Target | Printed ArUco marker (included in repo) |
| Host PC | Ubuntu 20.04 with ROS Noetic |

---

## Software Dependencies

This project runs inside a **Docker container** — see `HOW TO RUN PYTHON CODE ON THE UR3 - Docker.txt` for the full setup guide.

Core dependencies:

- **ROS Noetic**
- **Python 3.8+**
- **OpenCV** with ArUco module (`cv2.aruco`)
- **NumPy**
- **pyrealsense2** (Intel RealSense SDK)
- **cv_bridge** (ROS ↔ OpenCV)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## How It Works

### Eye-in-Hand Configuration

The RGB-D camera is bolted rigidly to the UR3 flange. The camera frame moves with the end-effector — the camera *is* the control frame, so no separate hand-eye calibration transform is required in the servoing loop.

```
UR3 Base → Joint 1–6 → End-Effector → [RGB-D Camera]
                                             ↓
                                     Sees ArUco marker
                                             ↓
                               Compute image feature error
                                             ↓
                               Send velocity commands to UR3
```

### Visual Servoing Pipeline

1. **Marker Detection** — Each camera frame is processed using `cv2.aruco.detectMarkers()` to locate the ArUco marker corners in pixel space.

2. **Feature Error** — The image features `s` (marker corner coordinates or centroid) are compared against the desired features `s*` (the target configuration):

   ```
   e = s − s*
   ```

3. **Interaction Matrix** — The interaction matrix `L` (image Jacobian) relates the time derivative of image features to the camera's 6-DOF velocity. Depth `Z` from the RGB-D channel populates the depth-dependent terms.

4. **Velocity Control Law** — The standard IBVS control law computes the corrective end-effector velocity:

   ```
   vc = −λ · L⁺ · e
   ```

   Where `λ` is a positive gain and `L⁺` is the Moore-Penrose pseudo-inverse of `L`.

5. **Robot Command** — The computed velocity is sent to the UR3 over ROS, driving the arm until `e → 0`.

---

## Running the System

> **Prerequisites:** The UR3 must be powered on, in Remote Control mode, and reachable on the network. The RGB-D camera must be mounted on the end-effector and connected via USB.

### Step 1 — Docker setup

Follow the full instructions in `HOW TO RUN PYTHON CODE ON THE UR3 - Docker.txt` to build and enter the Docker container.

### Step 2 — Initialise the UR3

```bash
./ur3_spinup
```

This establishes the ROS connection to the robot controller.

### Step 3 — Build the ROS workspace

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Step 4 — Print the ArUco target

Print `marker_0.png` (or `a4_markers_square.png` for a full sheet) at full scale and place it in the robot's workspace within the camera's field of view.

### Step 5 — Run the visual servoing node

```bash
rosrun <package_name> <main_script>.py
```

The robot will begin moving to align the camera with the detected ArUco marker. Motion stops when the feature error converges below the threshold.

---

## Visual Target

The ArUco markers in this repo use the `DICT_ARUCO_ORIGINAL` dictionary. Print at full scale — do not resize.

| File | Marker ID |
|------|-----------|
| `marker_0.png` | ID 0 |
| `marker_1.png` | ID 1 |
| `marker_2.png` | ID 2 |
| `a4_markers_square.png` | A4 sheet (all markers) |

---

## Demo Videos

| File | Description |
|------|-------------|
| `camera_detection.mp4` | RGB-D camera detecting an ArUco marker in real time |
| `1000010793.mp4` | Full visual servoing demonstration on the physical UR3 |

---

## Tuning Notes

- The gain `λ` controls convergence speed. Too high and the robot may overshoot; too low and convergence is slow. Tune empirically starting from a small value (e.g. `λ = 0.3`).
- Ensure adequate, consistent lighting on the ArUco marker for reliable detection.
- Network IP addresses and robot configuration in the launch/spinup scripts may need updating for environments outside the UTS CAS lab.

---

## License

MIT License © 2025 Abhi Naglapura
