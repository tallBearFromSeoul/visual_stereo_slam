Copyright [2022] [Gunn Lee]
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
# visual_stereo_slam
I wrote a C++ application of visual Simultaneous Localization And Mapping (SLAM) with obstacle detection neural net trained myself with a custom dataset using LibTorch under the inspiration from Tesla Autopilot.
Main libraries used : Eigen 3.3, g2o 1.0.0, OpenCV 4.5.5, LibTorch 1.10.0, CUDA 11.3, Pangolin 0.7

## My profile
https://www.linkedin.com/in/gunn-lee/

# Overview:
1. Demonstration video with key functionalities 
2. A brief description of the flow of the application
3. Prerequisites required to build and dependencies for building.

## Demonstration video with key functionalities
 ##### Visual Stereo Slam Demonstration Video
 ![visual_stereo_slam_demo](https://user-images.githubusercontent.com/97034587/165011639-47ab732a-99b3-40d4-b44e-8a2351bf89d4.gif)
 
 Full video of the above .gif file is at the link below :
       https://youtu.be/8iTiYWXvHkA

## A brief description of the flow of the application.
 ###### Functionality
   The main purpose of the application is to create a global map of the scene inputted from a series of images or a video of a driving scene while detecting pedestrians and vehicles observed from the camera. The application creates a 3D scene reconstructed from the input source composed of all frames’ pose, the points observed from each frame, and obstacles’ location. There are three functions to this application: - SLAM, -GPU_SLAM, GPU_STEREO_SLAM. The description below will be by default based on GPU_STEREO_SLAM. When using object detection module, the inference time is approximately around 8~9 seconds then delivers a stable speed of 0.19 sec per frame pair.
  ##### DETR
   Initially a frame pair is created, which preprocesses both image into tensors then feeds them into the DETR neural net. The DETR architecture is developed by Meta (Facebook at the time) where the features obtained from ResNet50 backbone are encoded and decoded with transformers, in which multi-head self-attention allows an efficient comparison of faraway pixels in a relatively big boundary box. The transformers output decoded n amount of boundary boxes of obstacles detected from ResNet50. During training, bipartite matching is performed with n amount of queries to ensure no duplicate detection and if there is no object in the boundary box, it matches to “no class”. 
  ##### DETR custom training
   The DETR is trained myself in a custom dataset of road driving images called “Udacity Self Driving Dataset” from Roboflow, where vehicles, pedestrians, and traffic lights are marked with boundary boxes in COCO format. The model was trained in Python for 500 epochs, which took around 6 days with the NVIDIA RTX GeForce 3090.
  ##### Stereo disparity matching
   The disparity map is obtained from a semi-global matching (SGM) algorithm (single camera version doesn’t include disparity map). Each pixel with a disparity value will be instanced as a Point then triangulated to estimate depth. All the points added from stereo disparity matching are rendered to the scene along with the pose of each frame.
  ##### Obstacles matching and triangulation
   Once the stereo disparity matching is over, every obstacle obtained from each frame of the pair is matched with the other pair’s. With the disparity value of the pixel in the boundary boxes’ centroids are used to triangulate the matched obstacle in both frame pair. The coordinates of the centroid of the boundary box is rendered to the scene with white concentric circles of different radius, while the latest pose of the frame pair is surrounded by cyan concentric circles of different radius, shown in Figure 1.
  ##### Pose estimation
   After the initialization, the next image pair is fed through DETR when creating FramePair. The obstacles are matched, then before the stereo disparity matching, the pose of the new frame pair is estimated through essential matrix through RANdum SAmple Consensus (RANSAC). When obtaining keypoints from each frame, Oriented FAST and Rotated BRIEF (ORB) feature detection and description is used to compute partial scale invariant features then expressed with BRIEF descriptors. The descriptors are compared and matched through k-nearest neighbors matched with k=2 and Lowe's ratio test with ratio of above 95%. The ransac algorithm is employed to reject outliers from the matches to obtain an essential matrix, which is used to extract R and t for pose.
  ##### Add already observed points in previous frame pair
   When matching the ORB keypoints between the nth and (n-1)th frame pair, if the keypoint is already a point in (n-1)th frame pair, then the corresponding keypoint in nth frame pair is added as a point in nth frame pair.
  ##### Local optimization
   After the pose is estimated, there is a local optimization of a new frame pair with its points through a sparse optimizer using g2o. The frame and points are added as vertices and all points are connected to their frame as edges with the transformation in between enforced as the constraint. 
  ##### Optimization in manifold
   When optimizing, all vertices into a Lie Group exponential map where SE(3) transformation becomes a 6 dimensional manifold. The constraints are enforced by the edges between the vertices. It is ideal to optimize SE(3) transformation on manifolds since rotation can be treated as a linear Lie group since it is a subgroup of GL(3,R). The vertices are mapped on the exponential map then with Levenberg-Marquardt iterate until error (the cost, which equals the logarithm map of the two SE(3) pose doesn't decrease anymore or exceeds max iteration.
  ##### Triangulation of inlier keypoints
   Once the frame pair is optimized, all the matched inlier keypoints between nth and (n-1)th frame are projected and triangulated. The points with reprojection error smaller than the threshold will be added as a Point.
  ##### Sliding window global optimization
   Before all the results are logged, every 5 frame pairs, there is a global sliding window optimization of 30 frames, or if n < 30, then n amount of frames. Similarly to local frame pair optimization, all the frames and their points are added to the optimizer and optimized. 
   ##### Keyframes   
   A frame pair becomes a key frame pair once it is outside the threshold boundary of the previous keyframe pair. And the key frames are highlighted relative to normal frames as shown below in Figure 2.
  ##### Rendering
  After the process_frames_cuda() method, draw_map() method is called in Display() class. The rendering is based on OpenGL and Pangolin, which renders all the frames as either dark red pyramids for default frame pair, or bright red and green each for keyframe pair. The frames are rendered as square pyramids, where the square vertex represents the origin and the vector from the origin to square’s center is the direction where the camera’s pointed at. The points are rendered with the original pixel value from the image. 
##### [Figure 1.0]
<img src=samples/img_1_0.png width="200" height="200" />

     Cyan concentric circle representing the latest frame pair's location.
##### [Figure 1.1]
<img src=samples/img_1_1.png width="200" height="200" />

      White concentric circle representing the location of matched and corrected obstacle. 
##### [Figure 2.]
<img src=samples/img_2.png width="150" height="150" />

    (UP) Default frame pair (DOWN) Keyframe pair
    Both looked from above, thus the projected side looks like a triangle.
The amount of points added from triangulation of inlier keypoints takes up about 80% of new points, stereo disparity and add from previous frame pair both shares 10% most of the time.

## Prerequisites required to build and dependencies for building.
Cmake minimum 3.0
C++17 used 
tested in Ubuntu Linux 18.04.06 LTS, NVIDIA GeForce RTX 3090, AMD Ryzen 5800X, 16Gb RAM.
Dependencies : 
``Eigen 3.3, g2o 1.0.0, OpenCV 4.5.5, LibTorch 1.10.0, CUDA 11.3, Pangolin 0.7``

