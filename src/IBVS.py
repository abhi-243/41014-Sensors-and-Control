import cv2
import numpy as np
import pyrealsense2 as rs

#=== PUBLIC SERVICE ANNOUNCEMENT ===
# to see wtf the formulas are check out vid three in:
# https://canvas.uts.edu.au/courses/36549/pages/5-summary-of-visual-servoing?module_item_id=2207438

#=== USE ===
# 1) create the class IBVS_Controller and set up pyrealsense 2 in main script
# 2) call calibrate_target_features while holding the board infront of the robot to save the desired target position
# 3) call run in a loop to receive joint velocities 


class IBVS_Controller():
    def __init__(self, robot_jacobian, proportional_constant, camera_profile, featureCount):

        # === pull camera specific variables ===

        color_stream = camera_profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.fl = np.array([intrinsics.fx, intrinsics.fy])  # focal length in pixels
        self.Cu, self.Cv = intrinsics.ppx, intrinsics.ppy        # principal point

        # === establish class variables ===

        self.Z = 50 # Depth estimation (Change to use deapth from camera)

        self.lambda_ = proportional_constant # about 0.1  Proportional constant (Changes how tight the control loop grips your balls) 0 -> 1

        self.imgJacobians = [] # Used to calculate the velocities (linear and angular) of the CAMERA. (not the arm yet, calm your horses fuck me)

        self.targetFeatures = [] # 2D coordnates of target position poses in format ([u1,v1,1], ... [u1,v1,1]) MUST BE SAME NUMBER AS FEATURES COMING OUT OF 'feature_extraction'

        self.robotJacobian = robot_jacobian

        self.numFeatures = featureCount

    def calibrate_target_features(self, image): # sets the target points based of current image
        try:
            ft = self.feature_extraction(image, self.numFeatures)
            self.targetFeatures = ft
            print(f"Target features saved: {ft}")
        except ValueError:
            print("WARNING: no features detected, target features unaltered")

    def run(self, rgb_image): # executes propper calculation sequence 
        f = self.feature_extraction(rgb_image, self.numFeatures)
        if len(f) != len(self.targetFeatures):
            raise ValueError("Target and current feature counts do not match.")
        jc = self.compute_image_jacobians(f)
        vc = self.calculate_camera_velocities(jc, f)
        q_velocities = self.calculate_joint_velocities(vc)

        return q_velocities

    def feature_extraction(self,image, numFeatures): # extracts features from a rgb frame
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(grayscale,maxCorners=numFeatures, qualityLevel=0.1, minDistance=10, useHarrisDetector=True)
        if corners is not None:
            corners = corners.reshape(-1, 2)
        else:
            raise ValueError("No features found.")
        return corners
    
    def compute_image_jacobians(self, features): 

        jacobians = [] # camera jacobians

        for i, feature in enumerate(features): # for every feature we detect, create a camera jacobian matrix
            u,v = feature
            fx,fy = self.fl
            uc = u - self.Cu
            vc = v - self.Cv
            jacobians.append(np.array([
                                        [-fx/(self.Z* self.Cu), 0, uc/self.Z, (self.Cu*uc*vc)/fx, -((fx**2) + (self.Cu**2) * (uc**2))/ (self.Cu * fx), vc],
                                        [0,-fy / (self.Cv * self.Z), vc/self.Z,((fy**2) + (self.Cv**2) * (vc**2)) / (self.Cv * fy), -(self.Cv*uc*vc)/fy,-uc]
                                       ]))
            
        return jacobians # camera jacobians
    
    def calculate_camera_velocities(self, imageJacobians, currFeatures):
        v = np.zeros((6,1)) # stores v_x, v_y, v_z, w_x, w_y, w_z of the camera
        psudoInverseJacobians = [] # fucked if i know bro, just trust
        J_total = np.vstack(imageJacobians) 
        J_pinv = np.linalg.pinv(J_total)    

        target = np.array(self.targetFeatures).reshape(-1, 2)
        error = (target - currFeatures).reshape(-1, 1)

        v = self.lambda_ * J_pinv @ error 

        return v
    
    def calculate_joint_velocities(self, velocities):
        inverseRobotJacobian = np.linalg.pinv(self.robotJacobian)

        q_velocities = inverseRobotJacobian @ velocities

        return q_velocities

