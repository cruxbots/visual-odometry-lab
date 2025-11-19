"""
Visual Odometry Implementation using Monocular Camera

This module implements a basic monocular visual odometry pipeline using:
- ORB feature detection and description
- Brute force feature matching
- Essential matrix estimation via RANSAC
- Pose recovery from essential matrix
- Trajectory accumulation and visualization

The pipeline processes consecutive image pairs from a sequence (e.g., KITTI dataset)
to estimate camera motion and reconstruct the 3D trajectory.

Note: Monocular visual odometry suffers from scale ambiguity. The estimated
trajectory is up to an unknown scale factor.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


def relevant_images(
    img_dir: Path = Path("~/datasets/kitti/kitti_format/data_scene_flow_multiview/training/image_2/").expanduser()
) -> List[Path]:
    """Load and sort image file paths from a directory.

    Args:
        img_dir: Path to directory containing image files. Defaults to KITTI dataset path.

    Returns:
        Sorted list of Path objects pointing to image files in the directory.
    """
    img_list: List[Path] = [x for x in img_dir.iterdir()]
    return sorted(img_list)

def feature_detect_orb(
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
    orb = cv2.ORB_create()
    # Convert to grayscale (ORB works on single-channel images)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    return kp1, des1, kp2, des2

def bf_feature_matching(
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
    # Hamming distance is appropriate for binary descriptors like ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match des2 (query) against des1 (train)
    matches = bf.match(des2, des1)
    # Sort by distance - lower distance = better match
    return sorted(matches, key=lambda x: x.distance)

def get_pose(
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
    # Camera intrinsic matrix for KITTI dataset (left camera, sequence 00)
    # Format: [fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0]
    P0 = np.array([
        9.842439e+02, 0.0, 6.900000e+02, 0.0,
        0.0, 9.808141e+02, 2.331966e+02, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]).reshape(3, 4)
    K = P0[:3, :3]  # Extract 3x3 intrinsic matrix

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
        pts1, pts2, K, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0
    )
    
    # Filter inliers (points that satisfy the essential matrix constraint)
    pts1_inlier = pts1[mask.ravel() == 1]
    pts2_inlier = pts2[mask.ravel() == 1]
    
    # Recover rotation and translation from essential matrix
    # Returns: number of inliers, rotation matrix, translation vector, inlier mask
    _, R, t, mask = cv2.recoverPose(E, pts1_inlier, pts2_inlier, K)
    
    return R, t

def plot_trajectory(trajectory: np.ndarray) -> None:
    """Visualize the estimated 3D camera trajectory.

    Creates a 3D plot showing the camera path through space. The trajectory
    is displayed as a continuous blue line, with the starting point marked
    in green and the ending point in red.

    Args:
        trajectory: Nx3 numpy array where each row is [x, y, z] position
                of the camera at a given time step.

    Note:
        The trajectory scale is arbitrary due to monocular scale ambiguity.
        Use ax.set_xlim/ylim/zlim to adjust viewing bounds if needed.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory as continuous line
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'b-', label='Trajectory', linewidth=2)
    
    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
            'go', label='Start', s=100)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
            'ro', label='End', s=100)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Uncomment to set fixed axis limits for better visualization
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-40, 40)
    # ax.set_zlim(-40, 40)
    
    ax.legend()
    ax.set_title('Visual Odometry Trajectory')
    plt.show()

def main_kitti(img_path_list: List[Path]) -> None:
    """Main visual odometry pipeline.

    Processes a sequence of images to estimate camera trajectory:
    1. For each consecutive image pair:
    - Detect and match features
    - Estimate relative pose (R, t)
    - Accumulate pose to get global trajectory
    2. Visualize the final trajectory

    The pose accumulation follows:
        R_global = R_global @ R        (compose rotations)
        t_global = t_global + R_global @ t  (transform translation to global frame)

    Args:
        img_path_list: List of paths to image files in sequence order.

    Note:
        - Press any key to advance to next frame during visualization
        - The trajectory is accumulated incrementally, so errors compound over time
        - Scale ambiguity means the trajectory scale is arbitrary
    """
    # Initialize global pose (identity at start)
    R_global = np.eye(3)  # Global rotation matrix (starts as identity)
    t_global = np.zeros((3, 1))  # Global translation vector (starts at origin)
    trajectory = []  # Store accumulated positions
    
    i = 0
    while i < len(img_path_list) - 1:
        # Load consecutive image pair
        img1 = cv2.imread(str(img_path_list[i]))
        img2 = cv2.imread(str(img_path_list[i + 1]))
        
        # Feature detection and description
        kp1, des1, kp2, des2 = feature_detect_orb(img1, img2)
        
        # Feature matching
        matches = bf_feature_matching(des1, des2)

        # Visualize matches (top 30 best matches)
        img_comp = cv2.drawMatches(
            img2, kp2,
            img1, kp2,
            matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        
        # Estimate relative pose between consecutive frames
        R, t = get_pose(matches, kp1, kp2)

        # Accumulate pose to get global trajectory
        # R_global: rotation from world frame to current camera frame
        R_global = R_global @ R
        
        # t_global: position of camera in world frame
        # Transform relative translation to global frame and add
        t_global = t_global + R_global @ t
        
        # Store current position
        trajectory.append(t_global.flatten())
        
        # Display matches (press any key to continue)
        cv2.imshow('Matched', img_comp)
        # Uncomment to plot trajectory incrementally:
        # plot_trajectory(np.array(trajectory))
        cv2.waitKey(0)
        
        i += 1
    
    cv2.destroyAllWindows()

    # Final trajectory visualization
    trajectory = np.array(trajectory)
    plot_trajectory(trajectory)

if __name__ == '__main__':
    """Entry point: Load images and run visual odometry pipeline."""
    # Load image sequence from default KITTI dataset path
    img_list = relevant_images()
    
    # Process first 40 images (adjust as needed)
    # Using fewer images helps debug and reduces computation time
    main_kitti(img_list[:40])