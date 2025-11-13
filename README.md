# Perception-Based Odometry Lab

A ROS 2 Python package for learning and experimenting with perception-based odometry techniques, including visual odometry, depth odometry, and visual-inertial odometry (VIO).

## Overview

This repository serves as a learning platform for implementing and understanding various odometry algorithms using perception sensors. The package is designed to work with mobile robots in both real-world and simulation environments.

## Current Status

🚧 **Work in Progress** - This is an active learning project.

Currently implements:
- Basic ROS 2 node for visual odometry
- Image subscription and processing pipeline

## Future Plans

- [ ] Monocular visual odometry
- [ ] Depth-based odometry
- [ ] Visual-Inertial Odometry (VIO)
- [ ] Integration with mobile robot platforms (real and simulated)

## Requirements

- ROS 2 (tested with Humble/Humble Hawksbill)
- Python 3.8+
- OpenCV (cv_bridge)
- rclpy

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

Run the visual odometry node:
```bash
ros2 run visual_odom_python vo_python
```

## Package Structure

```
visual_odom_python/
├── visual_odom_python/
│   ├── __init__.py
│   ├── visual_odom_python.py  # Main visual odometry node
│   └── notebooks/              # Jupyter notebooks for experimentation
├── test/                       # Unit tests
├── package.xml                 # ROS 2 package manifest
└── setup.py                    # Python package setup
```

## License

TODO: Add license information

## Maintainer

Rahul - mishrarahul1997@yahoo.com

