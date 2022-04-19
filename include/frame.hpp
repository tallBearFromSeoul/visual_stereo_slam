#include "kdTree.hpp"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudastereo.hpp>

#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>

class Point;
class Frame;
class Map;
class Log;

typedef std::shared_ptr<Frame> FramePtr;
typedef std::weak_ptr<Frame> wFramePtr;
typedef std::vector<wFramePtr> vec_wFramePtr;
typedef std::shared_ptr<Point> PointPtr;
typedef std::weak_ptr<Point> wPointPtr;
typedef std::vector<wPointPtr> vec_wPointPtr;

typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;

const std::vector<std::string> CLASSES = {"N/A", "biker", "car", "pedestrian", "trafficLight", "trafficLight-Green", "trafficLight-GreenLeft", "trafficLight-Red", "trafficLight-RedLeft", "trafficLight-Yellow","trafficLight-YellowLeft", "truck"};

class Obs {
	private:
		int _class;
		cv::Rect_<float> _bbox;
		Vec2d _center;
		Vec3d _depth3d;
	public:
		Obs(int class, cv::Rect_<float> __bbox, Vec2d __center, Vec3d __depth3d) : _bbox(__bbox), _center(__center), _depth3d(__depth3d);
		
		int class() {return _class;};
		cv::Rect_<float> bbox() {return _bbox;};
		Vec2d center() {return _center;};
		Vec3d depth3d() {return _depth3d;};

		double compare(Obs _obs) { return (_center - _obs.center()).squaredNorm();};

};

typedef std::vector<Obs> vec_Obs;

class Frame { 
	private:
		int _id;
		bool _is_cuda = false;
		bool _is_kf = false;

		Mat4d _pose;
		std::vector<cv::KeyPoint> _kps;
		//KDTree* kdTree();

		vec_Obs _obstacles;

		cv::Ptr<cv::FeatureDetector> _fdetector;
		cv::Ptr<cv::BFMatcher> _fmatcher;

		//cv::Ptr<cv::cuda::ORB> _fdetector_cuda;
		//cv::Ptr<cv::cuda::DescriptorMatcher> _fmatcher_cuda;

		torch::Tensor preprocess(const cv::Mat &_img);
		void detect_obstacles(const cv::Mat &, cv::Mat &, torch::jit::script::Module &);
		void extract_features(const cv::Mat &_img, const cv::Mat &_mask);
		void extract_features(const cv::cuda::GpuMat &_img);
		void anms(const int n);
		void normalize_kps(const Mat3d& _K_inv);

		double ratio_threshold = 0.95;//0.98;

	public:
		vec_wPointPtr pts;
		std::unordered_map<int, int> pts_idx_i;

		int obstacles_count() {return int(_obstacles.size());};
		vec_Obs obstacles() {return _obstacles;};
		vec_Obs::iterator obstacles_it() {return _obstacles.begin();};
		vec_Obs::iterator obstacles_end() {return _obstacles.end();};

		bool is_cuda() {return _is_cuda;};
		bool is_kf() {return _is_kf;};
		void set_kf() {_is_kf = true;};
		std::vector<cv::KeyPoint> kps() {return _kps;}
		void set_id(int __id) {_id = __id;};
		int id() {return _id;};
		bool isEmpty();
		bool getPointAtIdx(int, PointPtr &);

		Frame() {};
		Frame(const cv::Mat &_img, const Mat3d &_K_inv, const Map &);
		Frame(const cv::cuda::GpuMat &_img, const Mat3d &_K_inv, const Map &, const cv::Mat &, torch::jit::script::Module &);

		Frame(const cv::cuda::GpuMat &_img, const Mat3d &_K_inv, const Map &);

		//KDTree* tree;
		
		MatXd kpus_mat;
		MatXd kps_mat;
		cv::Mat desc;
		//cv::cuda::GpuMat desc_cuda;

		Mat4d getPose() {return _pose;};
		void setPose(const Mat4d& poseToSet) {_pose = poseToSet;};
		void setPoseIdentity() {_pose.setIdentity();};
		Vec3d getTrans() {return _pose.block(0,3,3,1);};

		void match_frames(const FramePtr &, std::vector<int> &_idx1, std::vector<int> &_idx2, std::vector<std::pair<Vec2d, Vec2d>> &_ret, Log *);
		MatXd triangulate(const FramePtr &, const std::vector<int>& idx1, const std::vector<int>& idx2);
};

class FramePair {
	private:
		int _id;
		bool _right_empty = true;
		cv::Mat _disparity;
		cv::Ptr<cv::StereoSGBM> _sgbm;
		cv::Ptr<cv::cuda::StereoSGM> _sgm_cuda;

	public:
		bool right_empty() {return _right_empty;};
		FramePtr left;
		FramePtr right;
		vec_wFramePtr lr;
		std::vector<int> idxL;
		std::vector<int> idxR;
		const Mat4d poseR {{0.9998543808844597, -0.01706309861700861, -0.00026017635946350924, -0.5094961871754736},
		{0.017064416377671962, 0.9998338346058513, 0.00641162000174109, -0.002022496204233391},
		{0.0001507310227716978, -0.006415126105036775, 0.9999794115066636, 0.005365297617411473}, 
		{0.0, 0.0, 0.0, 1.0}}; 
		
		FramePair(const cv::Mat &, const cv::Mat &, const Map &);
		FramePair(const cv::cuda::GpuMat &, const cv::cuda::GpuMat &, const Map &);
		FramePair(const cv::cuda::GpuMat &, const cv::cuda::GpuMat &, const Map &, const cv::Mat &, const cv::Mat &, torch::jit::script::Module &);

		void set_id(int __id) {_id = __id;};
		int id() {return _id;};
		
		void setPoseIdentity();
		void setRightPose();
		Vec3d getLeftTrans() {return left->getTrans();};

		void match_framePair(Log *);
		void disparity(const cv::Mat &, const cv::Mat &, Map &, const cv::Mat &, const cv::Mat &, Log *);
		void disparity(const cv::cuda::GpuMat &, const cv::cuda::GpuMat &, Map &, const cv::Mat &, const cv::Mat &, Log *);
		bool stereoDepth(const cv::KeyPoint &, const Mat3d &, Vec3d &);
		bool match_obstacles();
};

