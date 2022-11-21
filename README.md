# EKF_pose_estimation
Pose Estimation using an EKF to fuse wheel odometry and camera based pose estimation.

In this project, I am using Webots simulator for robot pose estimation. The robot has wheel encoders which provides wheel odomtery and a monocular camera which for pose estimation based on known landmarks in the environment with known global poses with respect to the world origin. Both these pose estimates are noisy (Gaussian noise). The pose estimation information from these sensors are fused using an  EKF in order to provide a robust robot localization.
