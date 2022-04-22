#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"
#include "essentialMat.hpp"
#include "eigen_utils.hpp"
#include "slam.hpp"
#include "log.hpp"
#include <opencv2/cudaimgproc.hpp>

void StereoSLAM::process_frame_cuda(cv::cuda::GpuMat &_imgcL_cuda, cv::cuda::GpuMat &_imgcR_cuda) {
	time_point<high_resolution_clock> now;
	duration<double> diff;
	now = high_resolution_clock::now();
	
	cv::Mat _imgcL, _imgcR;
	_imgcL_cuda.download(_imgcL);
	_imgcR_cuda.download(_imgcR);

	cv::cuda::GpuMat _imgL_cuda, _imgR_cuda;
	cv::cuda::cvtColor(_imgcL_cuda, _imgL_cuda, cv::COLOR_BGR2GRAY);	
	cv::cuda::cvtColor(_imgcR_cuda, _imgR_cuda, cv::COLOR_BGR2GRAY);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	now = high_resolution_clock::now();

	FramePairPtr F1 = std::make_shared<FramePair>(_imgL_cuda, _imgR_cuda, map, _imgcL, _imgcR, module);
	map.add_framePair(F1);
	if (F1->id() == 0) {
		F1->setPoseIdentity();
		if (F1->right_empty())
			F1->setRightPose();
		if (!F1->stereo_disparity(_imgL_cuda, _imgR_cuda, map, _imgcL, _imgcR, _log)) {
			assert(false);
			return;
		}
		map.optimize_map(2, false, false, true, opt_epochs, verbose_map, _log);
		map.add_keyFramePair(F1);
		return;
	}
	FramePairPtr F2 = map.map_framePairs[map.map_framePairs.size()-2];
	std::vector<int> inlier_idx1L, inlier_idx2L, inlier_idx1R, inlier_idx2R;
	
	// estimating pose
	now = high_resolution_clock::now();
	estimate_pose(F1->left, F2->left, inlier_idx1L, inlier_idx2L);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	
	// adding from previous framePair
	now = high_resolution_clock::now();
	add_from_prev_frame(F1->left, F2->left, inlier_idx1L, inlier_idx2L);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	// local optimization
	now = high_resolution_clock::now();
	double local_opt_error = map.optimize_map(2, false, false, true, opt_epochs, verbose_map, _log);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	// search by triangulation
	now = high_resolution_clock::now();
	search_by_triangulation(F1->left, F2->left, inlier_idx1L, inlier_idx2L, _imgcL);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	if (F1->right_empty())
		F1->setRightPose();

	// disparity map with SGM
	now = high_resolution_clock::now();
	F1->stereo_disparity(_imgL_cuda, _imgR_cuda, map, _imgcL, _imgcR, _log);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	// global optimization [SLDING WINDOW : 20 framePairs]
	now = high_resolution_clock::now();
	/*
	if (F1->left->id() >= global_opt_freq-1 && F1->left->id() % global_opt_freq == 0) {	
		double global_opt_error = map.optimize_map(-1, true, false, false, opt_epochs, verbose_map, _log);
		std::cout<<"Error from global optimization is "<<global_opt_error<<"\n";
	}
	*/
	if (relTransNorm(F1, map.map_keyFramePairs.back()) > keyFrame_threshold) {
		map.add_keyFramePair(F1);
		double global_opt_error = map.optimize_map(-1, true, false, false, opt_epochs, verbose_map, _log);
		std::cout<<"Error from global optimization is "<<global_opt_error<<"\n";
		//double global_kf_opt_error = map.optimize_map(-1, true, true, false, opt_epochs, verbose_map, _log);
		//std::cout<<"Error from global keyFrame optimization is : "<<global_kf_opt_error<<"\n";
	}
	mark_mapInfo(map.map_keyFrames.size(), map.map_keyFramePoints.size(), map.map_frames.size(), map.map_points.size());
}

void StereoSLAM::process_frame(const cv::Mat &_imgcL, const cv::Mat &_imgcR) { 
	time_point<high_resolution_clock> now;
	duration<double> diff;
	now = high_resolution_clock::now();

	cv::Mat _imgL, _imgR;
	cv::cvtColor(_imgcL, _imgL, cv::COLOR_RGB2GRAY);	
	cv::cvtColor(_imgcR, _imgR, cv::COLOR_RGB2GRAY);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	now = high_resolution_clock::now();

	FramePairPtr F1 = std::make_shared<FramePair>(_imgL, _imgR, map);
	map.add_framePair(F1);
	if (F1->id() == 0) {
		F1->setPoseIdentity();
		if (F1->right_empty())
			F1->setRightPose();
		F1->disparity(_imgL, _imgR, map, _imgcL, _imgcR, _log);
		// local optimization
		map.optimize_map(2, false, false, true, opt_epochs, verbose_map, _log);
		map.add_keyFramePair(F1);
		return;
	}
	FramePairPtr F2 = map.map_framePairs[map.map_framePairs.size()-2];
	std::vector<int> inlier_idx1L, inlier_idx2L, inlier_idx1R, inlier_idx2R;
	
	// estimating pose
	now = high_resolution_clock::now();
	estimate_pose(F1->left, F2->left, inlier_idx1L, inlier_idx2L);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	
	// adding from previous framePair
	now = high_resolution_clock::now();
	add_from_prev_frame(F1->left, F2->left, inlier_idx1L, inlier_idx2L);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	// local optimization
	now = high_resolution_clock::now();
	double local_opt_error = map.optimize_map(2, false, false, true, opt_epochs, verbose_map, _log);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	// search by triangulation
	now = high_resolution_clock::now();
	search_by_triangulation(F1->left, F2->left, inlier_idx1L, inlier_idx2L, _imgcL);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	if (F1->right_empty())
		F1->setRightPose();

	// disparity map with SGM
	now = high_resolution_clock::now();
	F1->disparity(_imgL, _imgR, map, _imgcL, _imgcR, _log);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	
	// global optimization [SLDING WINDOW framePairs]
	now = high_resolution_clock::now();
	if (F1->left->id() >= global_opt_freq-1 && F1->left->id() % global_opt_freq == 0) {	
		double global_opt_error = map.optimize_map(-1, true, false, false, opt_epochs, verbose_map, _log);
		std::cout<<"Error from global optimization is "<<global_opt_error<<"\n";
	}
	//std::cout<<"Latest KeyFrame pose : \n"<<map.map_keyFramePairs.back()->left->getPose()<<"\n";
	if (relTransNorm(F1, map.map_keyFramePairs.back()) > 3.0) {
		map.add_keyFramePair(F1);
		double global_kf_opt_error = map.optimize_map(-1, true, true, false, opt_epochs, verbose_map, _log);
		std::cout<<"Error from global keyFrame optimization is : "<<global_kf_opt_error<<"\n";
	}
	mark_mapInfo(map.map_keyFrames.size(), map.map_keyFramePoints.size(), map.map_frames.size(), map.map_points.size());
}

void SLAM::process_frame(const cv::Mat &_imgc) {
	cv::Mat _img;
	//detect_obstacles(_img, f1, map);
	cv::cvtColor(_imgc, _img, cv::COLOR_RGB2GRAY);	
	FramePtr f1 = std::make_shared<Frame>(_img, map);
	// initializing first frame
	map.add_frame(f1);
	if (f1->id() == 0) {
		f1->setPoseIdentity();
		return;
	}
	FramePtr f2 = map.map_frames[map.map_frames.size()-2];

	std::vector<int> inlier_idx1, inlier_idx2;
	estimate_pose(f1, f2, inlier_idx1, inlier_idx2);

	add_from_prev_frame(f1, f2, inlier_idx1, inlier_idx2);	

	// local optimization
	double local_opt_error = map.optimize_map(1, false, false, true, opt_epochs, verbose_map, _log);
	std::cout<<"Error from optimization is : "<<local_opt_error<<"\n";

	// search by projection
	if (!map.map_points.empty()) {
		search_by_projection(f1);
	}
	
	search_by_triangulation(f1, f2, inlier_idx1, inlier_idx2, _imgc);
	if (f1->id() >= global_opt_freq-1 && f1->id() % global_opt_freq == 0) {	
		int local_window = f1->id() < SLIDING_WINDOW ? f1->id()+1 : SLIDING_WINDOW;
		double global_opt_error = map.optimize_map(local_window, true, false, false, opt_epochs, verbose_map, _log);
		std::cout<<"Error from optimization is "<<global_opt_error<<"\n";
	}
	std::cout<<"There are "<<map.map_points.size()<<" amount of points and "<<map.map_frames.size()<<" amount of frames in map\n";
}

void SLAM::estimate_pose(const FramePtr &f1, const FramePtr &f2, std::vector<int> &inlier_idx1, std::vector<int> &inlier_idx2) {
	time_point<high_resolution_clock> now, t_begin;
	duration<double> diff;
	t_begin = high_resolution_clock::now();
	// matching frames
	std::vector<int> idx1, idx2;	
	std::vector<std::pair<Vec2d, Vec2d>> ret;
	f1->match_frames(f2, idx1, idx2, ret, log());
	diff = high_resolution_clock::now() - t_begin;
	mark_timing(diff);
	now = high_resolution_clock::now();

	EssentialMatrix EM;
	EM.ransac(ret, idx1, idx2, inlier_idx1, inlier_idx2);
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);
	now = high_resolution_clock::now();

	Mat4d pose4d = EM.poseFromE();
	diff = high_resolution_clock::now() - now;
	mark_timing(diff);

	mark_matchesInfo(inlier_idx1.size());
	// getting pose from previous pose;
	f1->setPose(pose4d * f2->getPose());
}

void SLAM::add_from_prev_frame(const FramePtr &f1, const FramePtr &f2, const std::vector<int> &inlier_idx1, const std::vector<int> &inlier_idx2) {
	int pts_added_prev_obs = 0;
	// add new observations if point has been observed in previous frame
	for (int i = 0; i<inlier_idx1.size(); i++) {
		if (f2->pts_idx_i.count(inlier_idx2[i]) && !f1->pts_idx_i.count(inlier_idx1[i])) {
			PointPtr temp = std::make_shared<Point>();
			if (f2->getPointAtIdx(inlier_idx2[i], temp)) {
				temp->add_observation(inlier_idx1[i], f1);
				pts_added_prev_obs++;
			} else {
				assert(false);
			}
		}
	}
	mark_ptsInfo(pts_added_prev_obs);
}

void SLAM::search_by_projection(const FramePtr &f1) {
	int sbp_pts_count = 0;
	// project all map points into current frame
	MatXd all_map_points(map.map_points.size(), 4);
	int map_point_idx = 0;
	for (const PointPtr &_pt : map.map_points){
		all_map_points.row(map_point_idx) = _pt->homogeneous();
		map_point_idx++;
	}
	// projecting
	// ((3x3 * 3x4) * Nx4.T).T -> Nx3
	int N = map.map_points.size();
	MatXd projs(N, 3);
	projs = ((K * f1->getPose().block(0,0,3,4)) * all_map_points.transpose()).transpose();
	// Nx2
	MatXd norm_projs(N, 2);
	norm_projs = projs.block(0,0,N,2).array().colwise() / projs.col(2).array();

	/*
	std::set<int> bad_norm_projs;
	check_outside_frame(norm_projs, W*W_inlier_ratio, H*H_inlier_ratio, bad_norm_projs);
	*/
	int map_point_idx2 = 0;
	for (const PointPtr &_pt : map.map_points){
		/*
		if (bad_norm_projs.count(map_point_idx2)) {
			map_point_idx2++;
			continue;
		}
		*/
		if (_pt->frames_ids.count(f1->id())) {
			map_point_idx2++;
			continue;
		}
		cv::KeyPoint pt_search_for(norm_projs.row(map_point_idx2)(0), norm_projs.row(map_point_idx2)(1), 20.0f);
		for (const size_t& idx: f1->tree->neighborhood_indices(pt_search_for, sbp_threshold)) {
			if (!f1->pts_idx_i.count(idx)) {
				double b_dist = _pt->orb_distance(f1->desc.row(idx));
				if (b_dist < b_dist_threshold) {
					_pt->add_observation(idx, f1);
					sbp_pts_count++;
				}
			}
		}
		map_point_idx2++;
	}
	delete f1->tree;
	std::cout<<"# points added by search by projection is : "<<sbp_pts_count<<"\n";
}


void SLAM::search_by_triangulation(const FramePtr &f1, const FramePtr &f2, const std::vector<int> &inlier_idx1, const std::vector<int> &inlier_idx2, const cv::Mat &imgc) {
	// triangulate all points initially
	MatXd pts4d = f1->triangulate(f2, inlier_idx1, inlier_idx2);
	MatXd pts4d_col3(pts4d.rows(), 1);
	pts4d_col3 = pts4d.block(0,3,pts4d.rows(),1);

	std::set<int> bad_pts4d;
  check_threshold_equal(pts4d_col3, 0.0, bad_pts4d);

	// turning into homogeneous coords
	pts4d.array().colwise() /= pts4d.col(3).array();

	// adding new points to map from pairwise matches
	int new_pts_count = 0;
	for (int i=0; i<int(pts4d.rows()); i++){
		if (bad_pts4d.count(i)) {
			continue;
		}
		// check if points are in front of cameras
		Vec4d pl1 = f1->getPose() * pts4d.row(i).transpose();
		Vec4d pl2 = f2->getPose() * pts4d.row(i).transpose();
		if (pl1(2) < 0 || pl2(2) < 0) {
			continue;
		}
		// reprojecting
		Vec3d pp1 = K * pl1.block(0,0,3,1);
	 	Vec3d pp2 = K * pl2.block(0,0,3,1);	
		pp1.block(0,0,2,1) /= pp1(2);
		pp2.block(0,0,2,1) /= pp2(2);
		Vec2d reproj_error1 = pp1.block(0,0,2,1).col(0).transpose().array() - f1->kpus_mat.row(inlier_idx1[i]).array();
		Vec2d reproj_error2 = pp2.block(0,0,2,1).col(0).transpose().array() - f2->kpus_mat.row(inlier_idx2[i]).array();
		if (reproj_error1.squaredNorm() > reproj_threshold || reproj_error2.squaredNorm() > reproj_threshold) {
			continue;
		}
		// add points
		cv::KeyPoint _kp = f1->kps()[inlier_idx1[i]];
		cv::Vec<uchar,3> p_c = imgc.at<cv::Vec<uchar,3>>(_kp.pt.y, _kp.pt.x);
		cv::Vec3f pixelColor = p_c;
		pixelColor/=255.0f;

		PointPtr pt = std::make_shared<Point>(pts4d.row(i).block(0,0,1,3).transpose(), pixelColor);
		map.add_point(pt);
		pt->add_observation(inlier_idx1[i], f1);
		pt->add_observation(inlier_idx2[i], f2);
		new_pts_count++;
	}
	mark_ptsInfo(new_pts_count);
}

void SLAM::reset_log() {
	_log->reset();
}

void SLAM::print_log(const int &_n, bool _verbose) {
	log()->print_log(_n, _verbose);
}

void SLAM::mark_timing(const duration<double> &_diff) {_log->mark_timing(_diff);};

void SLAM::mark_matchesInfo(const int &_I) {_log->mark_matchesInfo(_I);};

void SLAM::mark_ptsInfo(const int &_I) {_log->mark_ptsInfo(_I);};

void SLAM::mark_mapInfo(const int &_I0, const int &_I1, const int &_I2, const int &_I3) {
	_log->mark_mapInfo(_I0);
	_log->mark_mapInfo(_I1);
	_log->mark_mapInfo(_I2);
	_log->mark_mapInfo(_I3);
}


