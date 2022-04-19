#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <opencv2/core/types.hpp>

typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;
typedef Eigen::Matrix<double, 4, 4> Mat4d;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXd;

class EssentialMatrix {
	private:
		Mat3d E;
	
	public:

		EssentialMatrix(){E.setIdentity();};
		EssentialMatrix(const Mat3d &_E) {E = _E;};
		Mat4d poseFromE();
		void ransac(std::vector<std::pair<Vec2d, Vec2d>>& pts, const std::vector<int>& idx1, const std::vector<int>& idx2, std::vector<int>& inlier_idx1, std::vector<int>& inlier_idx2);
		bool estimate(const std::vector<std::pair<Vec2d, Vec2d>>& pts);
		MatXd residuals(const std::vector<std::pair<Vec2d, Vec2d>>& pts);
};	
