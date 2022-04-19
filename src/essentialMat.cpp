#include <random>
#include <iostream>
#include "eigen_utils.hpp"
#include "essentialMat.hpp"

Mat4d EssentialMatrix::poseFromE(){
	Mat4d res;
	Mat3d U, S, Vt, R;
	Mat3d W {{0,-1,0}, {1,0,0}, {0,0,1}};
	Vec3d t;
	Eigen::JacobiSVD<MatXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
	U = svd.matrixU();
	S = svd.singularValues().asDiagonal();
	Vt = svd.matrixV().transpose();
	if (U.determinant() < 0)
		U *= -1.0;
	if (Vt.determinant() < 0)
		Vt *= -1.0;
	R = (U*W)*Vt;
	if (R.diagonal().sum() < 0)
		R = (U*W.transpose())*Vt;
	t = U.block(0,2,3,1);
	if (t(2) < 0)
		t *= -1;
	res.setIdentity();
	res.block(0,0,3,3) = R;
	res.block(0,3,3,1) = t;
	return res.inverse();
}

void EssentialMatrix::ransac(std::vector<std::pair<Vec2d, Vec2d>>& pts, const std::vector<int>& idx1, const std::vector<int>& idx2, std::vector<int>& inlier_idx1, std::vector<int>& inlier_idx2){
	int num_samples = int(pts.size());
	int min_samples, max_trials, end_trials;
	double residual_threshold = 0.0007, stop_res_sum;
	if (num_samples > 300) {
		min_samples = num_samples/6;
		max_trials = 500;
		end_trials = 1000;
		stop_res_sum = 0.0012;
	} else if (num_samples > 50) {
		min_samples = 50;
		max_trials = 1000;
		end_trials = 4000;
		stop_res_sum = 0.0008;
	} else {
		assert(false);
	}

	std::vector<int> idxs;
	for (int i=0; i<pts.size(); i++){
		idxs.push_back(i);
	}
	
	std::random_device rd;
	std::mt19937 g(rd());

	int best_inliers_count = 0;
	double best_res_sum = std::numeric_limits<double>::max();
	std::vector<int> best_inliers;

	int num_trials = 0;
	while (num_trials < max_trials) {
		shuffle(idxs.begin(), idxs.end(), g);
		std::vector<std::pair<Vec2d, Vec2d>> samples;
		for (int i = 0; i<min_samples; i++){
			samples.push_back(pts[idxs[i]]);
		}

		bool success = estimate(samples);
		if (!success)
			continue;
		MatXd res = residuals(pts).array().abs();
		
		std::vector<int> inliers; 
		check_threshold_smaller(res, residual_threshold, inliers);
		double res_sum = res.squaredNorm();
		int inliers_count = int(inliers.size());
		if ((inliers_count > best_inliers_count) || (inliers_count == best_inliers_count && res_sum < best_res_sum)) {
			best_inliers_count = inliers_count;
			best_res_sum = res_sum;
			best_inliers = inliers;
		}
		num_trials++;
		if (best_res_sum < stop_res_sum)
			break;
		if (max_trials > end_trials)
			break;
		if (num_trials > max_trials-10)
			max_trials+=50;
	}
	std::vector<std::pair<Vec2d, Vec2d>> inlier_pts;
	if (!best_inliers.empty()){
		for (const int& inlier : best_inliers) {
			inlier_pts.push_back(pts[inlier]);
			inlier_idx1.push_back(idx1[inlier]);
			inlier_idx2.push_back(idx2[inlier]);
		}
		estimate(inlier_pts);
	}	
}


bool EssentialMatrix::estimate(const std::vector<std::pair<Vec2d, Vec2d>>& pts) {
	int N = int(pts.size());
	MatXd A(N, 9);
	MatXd Vt(9, 9);
	Mat3d F;
	// solving homogeneous eq : 
	// pts_2.T * F * pts_1 (Nx3 * 3x3 * 3xN)
 	A	= MatXd::Ones(N, 9);
	for (std::size_t i=0; i<N;i++){
		A(i,0) = pts[i].first(0);
		A(i,1) = pts[i].first(1);
	}	
	for (std::size_t i=0; i<N;i++){
		A.block(i,0,1,3) *= pts[i].second(0);
	}
	for (std::size_t i=0; i<N;i++){
		A(i,3) = pts[i].first(0);
		A(i,4) = pts[i].first(1);
	}
	for (std::size_t i=0; i<N;i++){
		A.block(i,3,1,3) *= pts[i].second(1);
	}
	for (std::size_t i=0; i<N;i++){
		A(i,6) = pts[i].first(0);
		A(i,7) = pts[i].first(1);
	}
	// obtaining nullspace of matrix A
	Eigen::JacobiSVD<MatXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);	
	Vt = svd.matrixV().transpose();
	// Because Eigen is column major 
	F = Vt.row(8).reshaped(3,3).transpose();
	// applying constraints for A
	Eigen::JacobiSVD<MatXd> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);	
	Mat3d U, S, Vt2;
	U = svd2.matrixU();
	S = svd2.singularValues().asDiagonal();
	S(0,0) = S(0,0)*S(1,1)*0.5;
	S(1,1) = S(0,0);
	S(2,2) = 0;
	Vt2 = svd2.matrixV().transpose();
	E = U*S*Vt2;
	return true;
}

MatXd EssentialMatrix::residuals(const std::vector<std::pair<Vec2d, Vec2d>>& pts) {
	int N = int(pts.size());
	MatXd A = MatXd(N, 3);
	MatXd B = MatXd(N, 3);
	MatXd E_At = MatXd(3, N);
	MatXd Et_Bt = MatXd(3, N);
	MatXd BE_Att = MatXd(N, 3);
	MatXd res = MatXd(N,1);
	Eigen::ArrayXXd denom = Eigen::ArrayXXd(N,1);
	for (std::size_t i=0; i<N;i++){
		A(i,0) = pts[i].first(0);
		A(i,1) = pts[i].first(1);
		A(i,2) = 1.0;
		B(i,0) = pts[i].second(0);
		B(i,1) = pts[i].second(1);
		B(i,2) = 1.0;
	}
	// E_At : 3xN
	E_At = E*A.transpose();
	// Et_Bt : Nx3
	Et_Bt = E.transpose()*B.transpose();
	// BE_Att : Nx1
	BE_Att = B.cwiseProduct(E_At.transpose()).rowwise().sum();
	// Computing Sampson distance
	denom = (E_At.row(0).array().square() + E_At.row(1).array().square() + Et_Bt.row(0).array().square() + Et_Bt.row(1).array().square()).sqrt();
	res = BE_Att.array().abs() / denom;	
	return res; 
}
