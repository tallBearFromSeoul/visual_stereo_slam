# visual_stereo_slam
I wrote a visual Simultaneous Localization And Mapping (SLAM).

Overview:
1. Introduction : A brief description of the functionality of the application.
2. Few demonstration purpose images and videos for visualization of the functionalities.
3. A detailed explanation of the algorithms employed and the flow of the application.
5. Additional information where each file's intention and purpose is recorded.
6. Prerequisites required to build and dependencies for building.

1. Introduction : A brief description of the functionality of the application.
  The main purpose of the application is to create a global map of the scene inputted 
from a series of images or a video of a driving scene while detecting all the vehicles
observed from the camera. The application creates a 3D scene reconstructed from the input 
series of images or video. With ALL mode in Display::draw_map() method, all the stereo frame 
pairs with stereo mode, or just all the frames. Initially a keyframe pair is created, where if 
a frame pair is no longer within the radius of the previous keyframe pair, that frame pair gets
registered as keyframe pair to the map.
  From any frame pair to a new frame pair, the ORB feature keypoints are computed then their 
descriptors are compared and matched through k-nearest neighbors match with k=2 and lowe's ratio
test with ratio of above 95%. Then from the matched keypoints. The RANdom SAmple Consensus (RANSAC) 
algorithm is employed to reject outliers from the matches to obtain essential matrix, used to 
obtain pose of the new frame pair.
  With stereo mode, the disparity map of the stereo image pair is obtained from semi-global matching 
(SGM) algorithm. All the points with disparity value will be added as a Point with the pixel values.
  All the points and frames are then inputted in the sparse optimizer, which minimizes the reprojection
errors between frames and points. So, all the points that are seen in a frame will have edge connected
from the point to the frame.



