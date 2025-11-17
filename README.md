# Perception-Based Odometry Lab

A ROS 2 Python package for learning and experimenting with perception-based odometry techniques, including visual odometry, depth odometry, and visual-inertial odometry (VIO).

## Overview

This repository serves as a learning platform for implementing and understanding various odometry algorithms using perception sensors. The package is designed to work with mobile robots in both real-world and simulation environments.

## Current Status

🚧 **Work in Progress** - This is an active learning project.

Currently implements:
- Basic ROS 2 node for visual odometry
- Image subscription and processing pipeline
- ✅ **Monocular visual odometry** - Standalone implementation with ORB features, essential matrix estimation, and trajectory visualization

## Future Plans

- [ ] Depth-based odometry
- [ ] Visual-Inertial Odometry (VIO)
- [ ] Integration with mobile robot platforms (real and simulated)
- [ ] Scale recovery for monocular VO
- [ ] Bundle adjustment and loop closure

## Requirements

- ROS 2 (tested with Humble/Humble Hawksbill) - *Optional, only for ROS 2 node*
- Python 3.8+
- OpenCV (cv_bridge for ROS 2, opencv-python for standalone)
- rclpy (for ROS 2 node only)
- NumPy
- Matplotlib (for trajectory visualization)

### Dataset

The standalone visual odometry implementation (`visual_odom_test.py`) is configured to work with the KITTI dataset. By default, it expects images at:
```
~/datasets/kitti/kitti_format/data_scene_flow_multiview/training/image_2/
```

You can modify the path in the `relevant_images()` function or pass a custom path.

## Installation

1. Clone this repository into your ROS 2 workspace:
```bash
cd ~/your_ws/src
git clone <repository-url> perception-odometry-lab
```

2. Install dependencies:
```bash
cd ~/your_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the package:
```bash
colcon build --packages-select visual_odom_python
source install/setup.bash
```

## Usage

### Standalone Monocular Visual Odometry

Run the standalone visual odometry implementation on image sequences:

```bash
cd ~/vo_ws/src/visual_odom_python
python3 -m visual_odom_python.visual_odom_test
```

Or directly:
```bash
python3 visual_odom_python/visual_odom_test.py
```

**Features:**
- ORB feature detection and matching
- Essential matrix estimation using RANSAC
- Relative pose recovery between consecutive frames
- 3D trajectory visualization
- Interactive frame-by-frame matching visualization

**Note:** The implementation processes consecutive image pairs and displays feature matches. Press any key to advance to the next frame. The final trajectory is plotted in 3D at the end.

### ROS 2 Node

Run the visual odometry ROS 2 node:
```bash
ros2 run visual_odom_python vo_python
```

## Package Structure

```
visual_odom_python/
├── visual_odom_python/
│   ├── __init__.py
│   ├── visual_odom_python.py  # ROS 2 visual odometry node
│   ├── visual_odom_test.py    # Standalone monocular VO implementation
│   └── notebooks/              # Jupyter notebooks for experimentation
│       └── odom_playbook.ipynb
├── test/                       # Unit tests
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── package.xml                 # ROS 2 package manifest
├── setup.py                    # Python package setup
└── README.md                   # This file
```

## Implementation Details

### Monocular Visual Odometry Pipeline

The `visual_odom_test.py` implements a complete monocular visual odometry pipeline:

1. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) features
2. **Feature Matching**: Brute force matcher with Hamming distance and cross-check validation
3. **Pose Estimation**: 
   - Essential matrix estimation via RANSAC
   - Rotation and translation recovery from essential matrix
4. **Trajectory Accumulation**: Incremental pose composition to build global trajectory
5. **Visualization**: 3D trajectory plotting with matplotlib

**Key Features:**
- Robust to outliers through RANSAC
- Handles point correspondence correctly
- Inlier filtering for better pose estimates
- Interactive visualization of feature matches

**Limitations:**
- Scale ambiguity (monocular VO cannot determine absolute scale)
- Errors accumulate over time (no loop closure or bundle adjustment)
- Requires sufficient texture and feature-rich scenes

## License

TODO: Add license information

## Maintainer

Rahul - mishrarahul1997@yahoo.com

