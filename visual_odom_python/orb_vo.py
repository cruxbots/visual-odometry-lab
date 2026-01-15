import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


class OrbVO:

    def __init__(
        self,
        intrinsic_param: Path = Path('visual_odom_python/config/rgbd_tum.txt')
        ) -> None:

        self.orb = cv2.ORB_create()
        # Hamming distance is appropriate for binary descriptors like ORB
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Format: [fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0]
        P0 = np.loadtxt(intrinsic_param).reshape(3,4)
        self.K = P0[:3,:3]
        self.prev_frame = None

    def feature_detect_orb(
        self,
        img1: np.ndarray, 
        img2: np.ndarray
        ) -> Tuple[List[cv2.KeyPoint], np.ndarray, List[cv2.KeyPoint], np.ndarray]:

        """Detect and compute ORB features and descriptors for two images.

        ORB (Oriented FAST and Rotated BRIEF) is a fast and robust feature detector
        suitable for real-time visual odometry applications. It combines the FAST
        keypoint detector with the BRIEF descriptor.

        Args:
            img1: First image (BGR format) as numpy array.
            img2: Second image (BGR format) as numpy array.

        Returns:
            Tuple containing:
                - kp1: List of keypoints detected in img1
                - des1: Descriptors for keypoints in img1 (numpy array)
                - kp2: List of keypoints detected in img2
                - des2: Descriptors for keypoints in img2 (numpy array)
        """
        
        # Convert to grayscale (ORB works on single-channel images)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        return kp1, des1, kp2, des2

    def bf_feature_matching(
        self,
        des1: np.ndarray, 
        des2: np.ndarray
        ) -> List[cv2.DMatch]:
        """Match feature descriptors using brute force matcher with Hamming distance.

        Important: The matching is done as bf.match(des2, des1), meaning:
        - query descriptors = des2 (from image 2)
        - train descriptors = des1 (from image 1)
        - In the resulting matches: queryIdx refers to des2/kp2, trainIdx refers to des1/kp1

        Args:
            des1: Descriptors from first image (train set).
            des2: Descriptors from second image (query set).

        Returns:
            List of DMatch objects sorted by distance (best matches first).
            Each match contains queryIdx (index in des2) and trainIdx (index in des1).
        """

        # Match des2 (query) against des1 (train)
        matches = self.bf.match(des2, des1)
        # Sort by distance - lower distance = better match
        return sorted(matches, key=lambda x: x.distance)

    def get_pose(
        self,
        matches: List[cv2.DMatch], 
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate relative pose (rotation and translation) between two camera views.

        This function implements the essential matrix approach for monocular visual odometry:
        1. Extract corresponding points from matched keypoints
        2. Estimate essential matrix using RANSAC
        3. Filter inliers based on RANSAC mask
        4. Recover rotation and translation from essential matrix

        The essential matrix E relates corresponding points in two views:
            x2^T * E * x1 = 0
        where x1 and x2 are normalized image coordinates.

        Important point correspondence fix:
        Since bf_feature_matching matches des2 (query) to des1 (train):
        - match.queryIdx -> index in kp2/des2 (image 2)
        - match.trainIdx -> index in kp1/des1 (image 1)

        Args:
            matches: List of matched feature correspondences from bf_feature_matching.
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.

        Returns:
            Tuple containing:
                - R: 3x3 rotation matrix representing rotation from image 1 to image 2
                - t: 3x1 translation vector (normalized, scale unknown) from image 1 to image 2

        Note:
            The camera intrinsic matrix K is hardcoded for KITTI dataset (left camera).
            Translation vector t is normalized (unit length) due to scale ambiguity in
            monocular vision. The actual scale must be recovered through other means
            (e.g., known motion, depth sensors, or scale recovery techniques).
        """

        # Extract corresponding points from matches
        # CRITICAL: Correct correspondence based on matching convention
        # trainIdx -> kp1 (image 1), queryIdx -> kp2 (image 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.queryIdx].pt for m in matches])

        # Estimate essential matrix using RANSAC
        # RANSAC parameters:
        #   - prob=0.999: Probability of finding at least one outlier-free sample
        #   - threshold=1.0: Maximum distance from epipolar line (in pixels)
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        # Filter inliers (points that satisfy the essential matrix constraint)
        pts1_inlier = pts1[mask.ravel() == 1]
        pts2_inlier = pts2[mask.ravel() == 1]
        
        # Recover rotation and translation from essential matrix
        # Returns: number of inliers, rotation matrix, translation vector, inlier mask
        _, R, t, mask = cv2.recoverPose(E, pts1_inlier, pts2_inlier, self.K)
        
        return R, t

    def vo_runner(self, frame):

        if self.prev_frame is None:
            self.prev_frame = frame
            return None, None

        curr_frame = frame
        prev_frame = self.prev_frame

        kp1, des1, kp2, des2 = self.feature_detect_orb(prev_frame,curr_frame)
        matches = self.bf_feature_matching(des1, des2)
        R, t = self.get_pose(matches, kp1, kp2)
        self.prev_frame = curr_frame
        return R,t

if __name__ == '__main__':

    orb_vo = OrbVO()