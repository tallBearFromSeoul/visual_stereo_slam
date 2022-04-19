#include <eigen3/Eigen/Dense>

template <typename T, int I, typename C>
inline void check_threshold_smaller(const Eigen::Matrix<T,I,I>& mat, T threshold, C& cont) {
	if (mat.size() == 0) return;
	for (size_t i=0, nRows=mat.rows(), nCols=mat.cols(); i<nRows; i++){
		for (size_t j=0; j<nCols; j++){
			if (mat(i,j) < threshold)
				cont.push_back(j);
		}
	}
}

template <typename T, int I, typename S>
inline void check_outside_frame(const Eigen::Matrix<T,I,I>& mat, T W, T H, S& set) {
	if (mat.size() == 0) return;
	for (size_t i=0, nRows=mat.rows(), nCols=mat.cols(); i<nRows; i++){
		for (size_t j=0; j<nCols; j++){
			if ((mat(i,j) <= 0.0) || (j == 0 && mat(i,j) > W) || (j == 1 && mat(i,j) > H)) {
				set.insert(i);
			}
		}
	}
}

template <typename T, int I, typename S>
inline void check_threshold_equal(const Eigen::Matrix<T,I,I>& mat, T threshold, S& set) {
	if (mat.size() == 0) return;
	for (size_t i=0, nRows=mat.rows(), nCols=mat.cols(); i<nRows; i++){
		for (size_t j=0; j<nCols; j++){
			if (mat(i,j) == threshold) {
				set.insert(i);
			}
		}
	}
}


