#include <set>
#include <unordered_set>
#include <eigen3/Eigen/Dense>
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <opencv2/calib3d.hpp>

class Log;
class Point;
class Frame;
class FramePair;

typedef std::shared_ptr<Point> PointPtr;
typedef std::vector<PointPtr> vec_PointPtr;
typedef std::shared_ptr<Frame> FramePtr;
typedef std::vector<FramePtr> vec_FramePtr;
typedef std::shared_ptr<FramePair> FramePairPtr;
typedef std::weak_ptr<FramePair> wFramePairPtr;
typedef std::vector<FramePairPtr> vec_FramePairPtr;
typedef std::vector<wFramePairPtr> vec_wFramePairPtr;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;

G2O_USE_OPTIMIZATION_LIBRARY(csparse);

class Map {
	private:
		//int OLD = 20;
		int VOLD = 100000;
		int SLIDING_WINDOW = 30;
		bool STEREO = false;

		int _max_point = 0;
		int _max_frame = 0;
		int _max_framePair = 0;

		Mat3d _K, _K_inv;

	public:
		cv::Ptr<cv::FeatureDetector> fdetector = cv::ORB::create(3000, 1.5f, 5, 31, 0);
		cv::Ptr<cv::BFMatcher> fmatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
		cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);

		//cv::Ptr<cv::cuda::ORB> fdetector_cuda = cv::cuda::ORB::create(3000, 1.5f, 5, 31, 0);
		//cv::Ptr<cv::cuda::DescriptorMatcher> fmatcher_cuda = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
		cv::Ptr<cv::cuda::StereoSGM> sgm_cuda = cv::cuda::createStereoSGM(0, 128, 8, 180, 3, cv::cuda::StereoSGM::MODE_HH);

		vec_PointPtr map_points;
		std::set<int> map_points_ids;
		vec_FramePtr map_frames;
		vec_FramePairPtr map_framePairs;
		
		vec_PointPtr map_keyFramePoints;
		vec_FramePtr map_keyFrames;
		vec_FramePairPtr map_keyFramePairs;

		Mat3d K() const {return _K;};
		Mat3d K_inv() const {return _K_inv;};

		Map() {};
		Map(const Eigen::Matrix3d& __K, bool _st=false) {_K=__K;_K_inv=__K.inverse();STEREO=_st;};
		
		int max_frame() {return _max_frame;};
		void add_point(const PointPtr &);
		void add_frame(const FramePtr &);
		void add_framePair(const FramePairPtr &);
		void add_keyFramePair(const FramePairPtr &);
		void add_keyFramePoints(const vec_wPointPtr &);

		Eigen::Vector3d normalize(const Vec2d& _kp);
		void cull(Log *);
		void refresh(Log *);
		void refresh_map(Log *);
		double optimize_map(int local_window, bool global, bool KF, bool fix_points, int rounds, bool _verbose, Log *);
		double optimize(const vec_FramePtr &, const vec_PointPtr &, int local_window, bool global, bool fix_points, int rounds, bool _verbose);
};

