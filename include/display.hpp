#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/types.hpp>

typedef Eigen::Vector3d Vec3d;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::Matrix4d Mat4d;
typedef std::shared_ptr<Point> PointPtr;
typedef std::vector<PointPtr> vec_PointPtr;
typedef std::shared_ptr<Frame> FramePtr;
typedef std::vector<FramePtr> vec_FramePtr;

class Map;

enum DrawType {
	ALL,
	KF_ONLY,
	HYBRID,
};

class Display {
	private:
		double x0 = 0;
		double y0 = 50;
		double z0 = 0.5;
		double p0 = 800;

		const std::string window_name;
		double w;
		double h;
		pangolin::OpenGlRenderState s_cam;
		
		void setup();
		void run(const std::pair<vec_FramePtr, vec_PointPtr> &);
	
	public:
		Display();
		Display(const double &_w, const double &_h);
		void DrawPoints(const vec_PointPtr &);
		void DrawCameras(const vec_FramePtr &);
		void DrawCamera(const Mat4d &, bool);
		void DrawObstacles(const std::vector<cv::Rect_<float>> &);
		void draw_map(const Map& map, DrawType);
};


