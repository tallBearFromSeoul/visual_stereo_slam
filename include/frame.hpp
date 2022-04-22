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

typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<float, 3, 3> Mat3f;
typedef Eigen::Matrix<float, 4, 4> Mat4f;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;

const std::vector<std::string> CLASSES = {"N/A", "biker", "car", "pedestrian", "trafficLight", "trafficLight-Green", "trafficLight-GreenLeft", "trafficLight-Red", "trafficLight-RedLeft", "trafficLight-Yellow","trafficLight-YellowLeft", "truck"};

class Obs {
	private:
		int _class_id = -1;
		//cv::Rect_<float> _bbox;
		float _area = -1.0f;
		Vec2f _w_h;
		Vec2f _center;
		Vec3f _depth3d;
	public:
		Obs(int __class_id, const Vec2f &__w_h, const Vec2f &__center) : _class_id(__class_id), _w_h(__w_h), _center(__center) {};
		//Obs(int __class_id, float __area, Vec2f __center) : _class_id(__class_id), _area(__area), _center(__center) {};
		//Obs(int __class_id, cv::Rect_<float> __bbox, Vec2f __center, Vec3f __depth3d) : _class_id(__class_id), _bbox(__bbox), _center(__center), _depth3d(__depth3d) {};
	
		int class_id() {return _class_id;};
		//cv::Rect_<float> bbox() {return _bbox;};
		float area() {return _area;};
		Vec2f w_h() {return _w_h;};
		Vec2f center() {return _center;};
		Vec3f depth3d() {return _depth3d;};
		
		void setDepth(const Vec3f &__depth3d) {_depth3d = __depth3d;};

		float compare(Obs *_obs_ptr) {float res=compare_w_h(_obs_ptr->w_h());res+=compare_center(_obs_ptr->center());return res;};
		float compare_w_h(const Vec2f &__w_h) {return (_w_h - __w_h).norm();};
		float compare_area(float __other_area) {return std::abs(_area - __other_area);};
		float compare_center(const Vec2f &__other_center) {return (_center - __other_center).norm();};

};

typedef std::vector<Obs> vec_Obs;

class Frame { 
	private:
		int _id;
		bool _is_cuda = false;
		bool _is_kf = false;

		Mat4d _pose;
		std::vector<cv::KeyPoint> _kps;
		vec_Obs _obstacles;
		std::vector<Obs *> _m_obs;

		cv::Ptr<cv::FeatureDetector> _fdetector;
		cv::Ptr<cv::BFMatcher> _fmatcher;
		//cv::Ptr<cv::cuda::ORB> _fdetector_cuda;
		//cv::Ptr<cv::cuda::DescriptorMatcher> _fmatcher_cuda;
		
		//torch::Tensor preprocess_image(const cv::Mat &_img);
		void extract_features(const cv::Mat &_img, const cv::_InputOutputArray &_mask=cv::noArray());
		void anms(const int n);
		void normalize_kps(const Mat3d& _K_inv);

		double ratio_threshold = 0.95;//0.98;

		//KDTree* kdTree();
		//KDTree* Frame::kdTree() {return new KDTree(_kps);};
	
	public:
		vec_wPointPtr pts;
		std::unordered_map<int, int> pts_idx_i;

		bool is_cuda() {return _is_cuda;};
		bool is_kf() {return _is_kf;};
		void set_kf() {_is_kf = true;};
		std::vector<cv::KeyPoint> kps() {return _kps;}
		void set_id(int __id) {_id = __id;};
		int id() {return _id;};
		bool isEmpty();
		bool getPointAtIdx(int, PointPtr &);

		void add_m_obs(Obs *_obs) {_m_obs.push_back(_obs);};
		void add_obstacle(const Obs &_obs) {_obstacles.push_back(_obs);};
		Frame() {};
		Frame(const cv::Mat &, const Map &);
		Frame(const cv::cuda::GpuMat &, const cv::Mat &, const Map &);
		//Frame(const cv::cuda::GpuMat &, const cv::Mat &, const Map &, torch::jit::script::Module &);

		KDTree* tree;
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

		torch::Tensor preprocess_image(const cv::Mat &_img);
		int obstacles_count() {return int(_obstacles.size());};
		Obs * obstacles(size_t _n) {return &_obstacles[_n];};
		vec_Obs obstacles() {return _obstacles;};
};

class FramePair {
	private:
		int _id;
		bool _right_empty = true;
		cv::Mat _disparity;
		cv::Ptr<cv::StereoSGBM> _sgbm;
		cv::Ptr<cv::cuda::StereoSGM> _sgm_cuda;
	
		std::vector<std::priority_queue<std::vector<float>, std::vector<std::vector<float>>, std::greater<std::vector<float>>>> _matches;

		std::vector<Vec3f> _corr_m_obs;

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

		std::vector<std::priority_queue<std::vector<float>, std::vector<std::vector<float>>, std::greater<std::vector<float>>>> matches() {return _matches;};
		std::vector<Vec3f> corr_m_obs() {return _corr_m_obs;};
		//std::pair<torch::Tensor> preprocess_imagePair(const cv::Mat &_img);
		void set_id(int __id) {_id = __id;};
		int id() {return _id;};
		
		void setPoseIdentity();
		void setRightPose();
		Vec3d getLeftTrans() {return left->getTrans();};

		void disparity(const cv::Mat &, const cv::Mat &, Map &, const cv::Mat &, const cv::Mat &, Log *);
		void disparity(const cv::cuda::GpuMat &, const cv::cuda::GpuMat &, Map &, const cv::Mat &, const cv::Mat &, Log *);
		bool stereoDepth(const cv::KeyPoint &, const Mat3d &, Vec3d &);
		
		bool stereo_disparity(const cv::cuda::GpuMat &, const cv::cuda::GpuMat &, Map &, const cv::Mat &, const cv::Mat &, Log *);
		
		template<typename D>
		void depth_estimate(Obs *, const FramePtr &, const Eigen::Matrix<D,3,3> &, const Eigen::Matrix<D,4,4> &);
		bool obstacles_depth_est(const Mat3d &);
		bool match_obstacles();
		void detect_obstacles(const FramePtr &, const std::vector<torch::jit::IValue> &_inputs, torch::jit::script::Module &_module, const Mat3d &, const Mat4d &, const cv::_InputOutputArray &_in_img=cv::noArray(), const cv::_InputOutputArray &_in_mask=cv::noArray());
		torch::Tensor preprocess_image(const cv::Mat &_img);
};

