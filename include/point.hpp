#include <set>
#include <unordered_map>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/types.hpp>

class Frame;
class Map;

typedef std::shared_ptr<Frame> FramePtr;
typedef std::weak_ptr<Frame> wFramePtr;
typedef std::vector<wFramePtr> vec_wFramePtr;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;

class Point : public std::enable_shared_from_this<Point> {
	private:
		int _id;
		cv::Vec3f _color;
		bool is_color = false;

	public:	
		std::unordered_map<int, int> f_id_idx;
		vec_wFramePtr frames;		
		std::set<int> frames_ids;

		Vec3d xyz;
		cv::Vec3f color() {return _color;};
		bool isColor() {return is_color;};
		void setColor(const cv::Vec3f &_c) {_color = _c;};
		
		Point() {xyz=Vec3d(0,0,0), _color=cv::Vec3f(255,0,0);};
		//Point(const Vec3d &_xyz) : xyz(_xyz) {_color = cv::Vec3f(255,0,0);};
		Point(const Vec3d &_xyz, const cv::Vec3f &_c) : xyz(_xyz), _color(_c) {is_color = true;};

		void set_id(int __id) {_id = __id;};
		int id() {return _id;};
		int latest_f_id();
		void add_observation(int idx, const FramePtr &);
		void delete_point();

		Vec4d homogeneous();
		double hamming_distance(const cv::Mat &a, const cv::Mat &b);
		double orb_distance(const cv::Mat &desc);
};
