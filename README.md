# Calibrating Football Fields with Keypoints
This repository implements a computer vision pipeline for calibrating football fields in video footage using keypoint detection. By leveraging object detection techniques, it maps key landmarks on the pitch (e.g., corners, center spot, penalty areas) to enable accurate player tracking, homography estimation, and sports analytics. 

The approach addresses challenges like camera movement, occlusions, and varying field conditions.
The project transforms single-pixel keypoints into bounding boxes for robust detection using YOLO models. It uses a labeled dataset from Roboflow with 317 frames and 32 classes of pitch features.



## Medium Blogs
The project is detailed in a Medium series by Erfan Akbarnezhad (@akbarnezhad1380):

### Part 1: Calibrating Football Fields with Keypoints (Part 1/3)

Published: July 26, 2025

[Link](https://medium.com/@akbarnezhad1380/calibrating-football-fields-with-keypoints-part-1-3-88fa4aad4d6e?source=user_profile_page---------1-------------56c1a23f082f----------------------)

Introduces the goal of pitch feature detection, dataset preparation, YOLOv11n training, and initial results with pixel-level accuracy.



### Part 2: Calibrating Football Fields with Keypoints (Part 2/3)
Published: July 29, 2025

[Link](https://medium.com/@akbarnezhad1380/calibrating-football-fields-with-keypoints-part-2-3-daea248585e0?source=user_profile_page---------0-------------56c1a23f082f----------------------)




Refer to the Medium articles for benchmarks and visuals.
