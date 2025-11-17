import numpy as np
import cv2
from pathlib import Path

def relevant_images(
    img_dir:Path = Path("~/datasets/kitti/kitti_format/data_scene_flow_multiview/training/image_2/").expanduser()
    )->list:

    img_list:list = [x for x in img_dir.iterdir()]
    return sorted(img_list)

def feature_detect_orb(img1:np.array,img2:np.array):
    """feature detections and description using orb

    Args:
        img1 (np.array): image 1
        img2 (np.array): image 2

    Returns:
        _type_: kp1, des1, kp2, des2
    """
    orb = cv2.ORB_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    return kp1, des1, kp2, des2

def bf_feature_matching(des1, des2) -> list:

    """simple brute force matcher

    Returns:
        _type_: sorted matches
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des2, des1)
    return sorted(matches, key=lambda x:x.distance)

def get_pose(matches, kp1, kp2):


    # Camera Matrix
    P0 = np.array([9.842439e+02, 0.0, 6.900000e+02, 0.0,
            0.0, 9.808141e+02, 2.331966e+02, 0.0,
            0.0, 0.0, 1.0, 0.0]).reshape(3,4)
    K = P0[:3, :3]

    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])  # trainIdx -> kp1
    pts2 = np.float32([kp2[m.queryIdx].pt for m in matches])  # queryIdx -> kp2

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    pts1_inliner = pts1[mask.ravel() == 1]
    pts2_inliner = pts2[mask.ravel() == 1]
    _, R, t, mask = cv2.recoverPose(E, pts1_inliner, pts2_inliner, K)
    return R, t

def plot_trajectory(trajectory):
    """Plot the 2D/3D trajectory"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 'go', label='Start', s=100)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 'ro', label='End', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-40, 40)
    # ax.set_zlim(-40, 40)
    ax.legend()
    ax.set_title('Visual Odometry Trajectory')
    plt.show()

def main(img_path_list:list[Path]):

    R_global = np.eye(3)
    t_global = np.zeros((3,1))
    trajectory = []
    i = 0
    while i < len(img_path_list) -1:

        img1 = cv2.imread(str(img_path_list[i]))
        img2 = cv2.imread(str(img_path_list[i+1]))
        kp1, des1, kp2, des2 = feature_detect_orb(img1, img2)
        matches = bf_feature_matching(des1,des2)

        img_comp = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        R, t = get_pose(matches, kp1, kp2)

        R_global = R_global @ R
        t_global = t_global + R_global @ t
        trajectory.append(t_global.flatten())
        cv2.imshow('Matched', img_comp)
        # plot_trajectory(np.array(trajectory))
        cv2.waitKey(0)
        i= i+1
    cv2.destroyAllWindows()

    trajectory = np.array(trajectory)
    plot_trajectory(np.array(trajectory))

if __name__ == '__main__':

    img_list = relevant_images()
    main(img_list[:40])