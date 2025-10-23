import cv2
import numpy as np
import pyrealsense2 as rs

class IBVS_Controller():
    """
    Modified IBVS controller:
    - Accepts explicit 2D feature centroids (u,v)
    - Supports per-feature depth (Zs) for image Jacobians
    - Allows setting target features explicitly (keeps order)
    """

    def __init__(self, robot_jacobian, proportional_constant, camera_profile, featureCount):
        color_stream = camera_profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.fl = np.array([intrinsics.fx, intrinsics.fy])  # focal length in pixels
        self.Cu, self.Cv = intrinsics.ppx, intrinsics.ppy    # principal point

        # default per-feature depths (will be overwritten each cycle)
        self.Zs = None

        self.lambda_ = proportional_constant
        self.imgJacobians = []
        self.targetFeatures = None   # explicit list of centroids [[u,v],[u,v],...]
        self.robotJacobian = robot_jacobian
        self.numFeatures = featureCount

    # set target features from explicit list of centroids (ordered)
    def set_target_features(self, centroids):
        if len(centroids) != self.numFeatures:
            raise ValueError("target centroid count mismatch")
        self.targetFeatures = np.array(centroids).reshape(-1, 2)
        print("IBVS: target features set:", self.targetFeatures.tolist())

    # compute image jacobians using per-feature Zs (list or array length numFeatures)
    def compute_image_jacobians(self, features, Zs=None):
        """
        features: Nx2 array of pixel coords [[u,v],...]
        Zs: either scalar or list/array length N in meters
        returns list of 2x6 jacobians (image interaction matrices)
        """
        if features is None:
            raise ValueError("No features provided")
        features = np.array(features).reshape(-1, 2)
        n = features.shape[0]
        if n != self.numFeatures:
            raise ValueError("Feature count mismatch")

        # handle Zs
        if Zs is None:
            # fallback to mean placeholder
            Zs = np.ones(n) * 0.5
        else:
            Zs = np.array(Zs).flatten()
            if Zs.size == 1:
                Zs = np.ones(n) * float(Zs)
            elif Zs.size != n:
                raise ValueError("Zs length mismatch")

        fx, fy = self.fl
        jacobians = []
        for i, feature in enumerate(features):
            u, v = feature
            z = float(Zs[i])
            uc = u - self.Cu
            vc = v - self.Cv

            # NOTE: kept your original matrix form but using z (meters)
            # Many IBVS derivations use fx/z etc. This follows your structure.
            J_i = np.array([
                [-fx/(z * self.Cu), 0, uc / z,
                 (self.Cu * uc * vc) / fx,
                 -((fx**2) + (self.Cu**2) * (uc**2)) / (self.Cu * fx),
                 vc],
                [0, -fy / (self.Cv * z), vc / z,
                 ((fy**2) + (self.Cv**2) * (vc**2)) / (self.Cv * fy),
                 -(self.Cv * uc * vc) / fy,
                 -uc]
            ], dtype=float)
            jacobians.append(J_i)
        return jacobians

    def calculate_camera_velocities(self, imageJacobians, currFeatures):
        """
        imageJacobians: list of 2x6 matrices for each feature
        currFeatures: Nx2 array of current pixel coords
        returns v_cam: 6x1 vector (camera twist)
        """
        if self.targetFeatures is None:
            raise RuntimeError("Target features not set")

        J_total = np.vstack(imageJacobians)  # (2N x 6)
        # In case we only control u,v (2D), we still compute full 6D twist but later caller may zero components
        J_pinv = np.linalg.pinv(J_total)

        target = np.array(self.targetFeatures).reshape(-1, 2)
        error = (target - np.array(currFeatures)).reshape(-1, 1)  # (2N x 1)

        v = self.lambda_ * (J_pinv @ error)  # 6x1
        return v

    def calculate_joint_velocities(self, velocities):
        """
        velocities: 6x1 twist (in whatever frame the robotJacobian maps to)
        robotJacobian: 6x6 jacobian mapping q_dot -> twist_in_same_frame
        returns q_dot (6x1)
        """
        if self.robotJacobian is None:
            raise RuntimeError("robotJacobian not set")
        qdot = np.linalg.pinv(self.robotJacobian) @ velocities
        return qdot