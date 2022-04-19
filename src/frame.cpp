#include <unordered_set>
#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include "log.hpp"

using namespace std::chrono;
using namespace torch::indexing;

bool FramePair::match_obstacles() {
	for (vec_Obs::iterator it = left->obstacles_it(); it != left->obstacles_end(); ) {
		cv::Rect_<float> bbox = it->bbox().area();
		int class_id = it->class();
		cv::Rect_<float> bbox = it->bbox();
		Vec2d center = it->center();
		Vec3d depth3d = it->depth3d();
		std::cout<<"class id : "<<class_id<<"\n";
		std::cout<<"bbox : "<<bbox<<"\n";
		std::cout<<"center : "<<center<<"\n";
		std::cout<<"depth3d : "<<depth3d <<"\n";
	}
}

torch::Tensor Frame::preprocess(const cv::Mat &_img) {
	torch::Tensor img_tensor = torch::from_blob(_img.data, {_img.rows, _img.cols, 3});//, options);
	
	img_tensor = img_tensor.permute({2,0,1});
	const float mean[3] = {0.485, 0.456, 0.406};
	const float std[3] = {0.229, 0.224, 0.225};
	for (int ch=0; ch<3; ch++) {
		img_tensor[ch].sub_(mean[ch]).div_(std[ch]);
	}
	//img_tensor.unsqueeze_(0);
	return img_tensor.clone();
}

void Frame::detect_obstacles(const cv::Mat &_img, cv::Mat &_mask, torch::jit::script::Module &_module) {
	cv::Mat _img_clone = _img.clone();
	torch::Tensor img_tensor = preprocess(_img);
	img_tensor = img_tensor.to(at::kCUDA);
	std::vector<torch::Tensor> images = {img_tensor};
	std::vector<int> img_size = {_img.rows, _img.cols};
	float w = float(_img.cols);
	float h = float(_img.rows);
	std::vector<torch::jit::IValue> inputs = {images, h, w};
	
	c10::Dict output = _module.forward(inputs).toGenericDict();
	torch::Tensor scores = output.at("scores").toTensorList()[0];
	torch::Tensor pred_classes = output.at("pred_classes").toTensorList()[0];
	torch::Tensor pred_boxes = output.at("pred_boxes").toTensorList()[0];
	
	for (int i=0; i<scores.sizes()[0]; i++) {
		int c = pred_classes.index({i}).item<long>();
		torch::Tensor box = pred_boxes.index({i, Slice()});
		float x1, x2, c_x, _x, y1, y2, c_y, _y;
		x1 = box[0].item<float>();
		y1 = box[1].item<float>();
		x2 = box[2].item<float>();
		y2 = box[3].item<float>();
		x1 = x1 < 0 ? 0 : x1;
		y1 = y1 < 0 ? 0 : y1;
		x2 = x2-x1 > w ? w-x1 : x2;
		y2 = y2-y1 > h ? h-y1 : y2;
		cv::Rect_<float> bbox(x1, y1, x2-x1, y2-y1);
		//cv::Rect_<float> bbox(y1, x1, y2-y1, x2-x1);

		c_x = x2-(x2-x1)/2;
		c_y = y2-(y2-y1)/2;
		std::cout<<"c_x , c_y : "<<c_x<<" , "<<c_y<<"\n";
		float val, _depth;
		val = _disparity.at<float>(c_x, c_y);
		_x = (c_x - _K(6)) / _K(0);
		_y = (c_y - _K(7)) / _K(0);
		_depth = K(0) / val;
		std::cout<<"depth : detect_obstacles : "<<_depth<<"\n";
		if (_depth < 0) {
			std::cerr<<"detect_obstacles assertion failure\n";
			assert(false);
		}
		Vec4d _res;
		_rel_depth_homo << _x*_depth, _y*_depth, _depth, 1.0;
		Vec3d depth3d = getPose().inverse().block(0,0,3,4)*_rel_depth_homo;
		std::cout<<"3d loc xyz : detect_obstacles : "<<depth3d<<"\n";
		Vec2d center(c_x, c_y);
		Obs obs(c, bbox, center, depth3d);
		_obstacles.push_back(obs);

		_mask(bbox) = 0;
		cv::rectangle(_img_clone, bbox, cv::Scalar(0,0,255), 2);
		cv::putText(_img_clone, CLASSES[c], cv::Point(x1, y1), 2, 1.2, cv::Scalar(0,255,0));
	}
	cv::imshow("img_clone", _img_clone);
	cv::waitKey(500);
}

FramePair::FramePair(const cv::cuda::GpuMat &_imgL_cuda, const cv::cuda::GpuMat &_imgR_cuda, const Map &_map, const cv::Mat &_imgcL, const cv::Mat &_imgcR, torch::jit::script::Module &_module) {
	left = std::make_shared<Frame>(_imgL_cuda, _map.K_inv, _map, _imgcL, _module);
	right = std::make_shared<Frame>(_imgR_cuda, _map.K_inv, _map, _imgcR, _module);
	lr = {left, right};
	_sgm_cuda = _map.sgm_cuda;
	
	std::cout<<"# of obstacles : [LEFT] "<<left->obstacles_count()<<"\t[RIGHT] "<<right->obstacles_count()<<"\n";
	if (!match_obstacles()) {
		std::cerr<<"match obstacle results false\n";
		assert(false);
	}
	// if the area of rectangles are similar and the obstacles count are the same if not more problems 
}

Frame::Frame(const cv::cuda::GpuMat &_img_cuda, const Mat3d &_K_inv, const Map &_map, const cv::Mat &_img, torch::jit::script::Module &_module) {
	cv::Mat _img_float, mask;
	mask = cv::Mat::ones(_img.rows, _img.cols, 0);
	_img.convertTo(_img_float, CV_32FC3, 1.0f/255.0f);
	
	time_point<high_resolution_clock> now;
	duration<double> diff;
	now = high_resolution_clock::now();
	
	detect_obstacles(_img_float, mask, _module);
	diff = high_resolution_clock::now() - now;
	std::cout<<"Time taken for detecting obstacles : "<<diff.count()<<"\n";

	_is_cuda = true;
	_fdetector = _map.fdetector;
	//_fdetector_cuda = _map.fdetector_cuda;
	_fmatcher = _map.fmatcher;
	//_fmatcher_cuda = (_map.fmatcher_cuda);
	extract_features(_img, mask);
	normalize_kps(_K_inv);
	//tree = kdTree();
}

Frame::Frame(const cv::cuda::GpuMat &_img_cuda, const Mat3d &_K_inv, const Map &_map) {
	_is_cuda = true;
	_fdetector = _map.fdetector;
	//_fdetector_cuda = _map.fdetector_cuda;
	_fmatcher = _map.fmatcher;
	//_fmatcher_cuda = (_map.fmatcher_cuda);
	extract_features(_img_cuda);
	normalize_kps(_K_inv);
	//tree = kdTree();
}

void Frame::extract_features(const cv::cuda::GpuMat &_img_cuda) {
	//_fdetector_cuda->detect(_img, _kps);
	//anms(1500);
	//_fdetector_cuda->compute(_img, _kps, desc_cuda);
	//desc_cuda.download(desc);
}

FramePair::FramePair(const cv::cuda::GpuMat &_imgL, const cv::cuda::GpuMat &_imgR, const Map &_map) {
	left = std::make_shared<Frame>(_imgL, _map.K_inv, _map);
	right = std::make_shared<Frame>(_imgR, _map.K_inv, _map);
	lr = {left, right};
	_sgm_cuda = _map.sgm_cuda;
}

void FramePair::disparity(const cv::cuda::GpuMat &_imgL, const cv::cuda::GpuMat &_imgR, Map &_map, const cv::Mat &_imgcL, const cv::Mat &_imgcR, Log *_log) {
	cv::cuda::GpuMat _disparity_cuda, _disparity_sgbm_cuda;
	_sgm_cuda->compute(_imgL, _imgR, _disparity_sgbm_cuda);
	_disparity_sgbm_cuda.convertTo(_disparity_cuda, CV_32F, 1.0 / 16.0f);
	_disparity_cuda.download(_disparity);
	
	int stereo_disparity_pts_count = 0;
	for (const wFramePtr &_wf : lr) {
		FramePtr _f = _wf.lock();
		if (!_f)
			continue;
		for (size_t idx=0; idx<=_f->kps().size()-1; idx++) {
			cv::KeyPoint _kp = _f->kps()[idx];
			Vec3d rel_depth;
			if (!stereoDepth(_kp, _map.K, rel_depth))
				continue;
			Vec4d rel_depth_homo;
			rel_depth_homo << rel_depth, 1.0;
			if (rel_depth(2) > 5 && rel_depth(2) < 500) {
				Vec3d kp_3d = _f->getPose().inverse().block(0,0,3,4)*rel_depth_homo;
				const cv::Mat* img_ptr;
				if (_f->id()%2==0) {
					img_ptr = &_imgcL;
				}	else {
					img_ptr = &_imgcR;
				}
				cv::Vec<uchar,3> p_c = img_ptr->at<cv::Vec<uchar,3>>(_kp.pt.y, _kp.pt.x);
				cv::Vec3f pixelColor = p_c;
				pixelColor/=255.0f;
				PointPtr pt = std::make_shared<Point>(kp_3d, pixelColor);
				_map.add_point(pt);
				pt->add_observation(idx, _f);
				stereo_disparity_pts_count++;
			}
		}
	}
	_log->mark_ptsInfo(stereo_disparity_pts_count);
}


Frame::Frame(const cv::Mat &_img, const Mat3d &_K_inv, const Map &_map) {
	_fdetector = _map.fdetector;
	_fmatcher = _map.fmatcher;

	cv::Mat mask = cv::Mat::ones(_img.rows, _img.cols, 0);
	extract_features(_img, mask);
	normalize_kps(_K_inv);
	tree = kdTree();
}

KDTree* Frame::kdTree() {
	return new KDTree(_kps);
}

void Frame::extract_features(const cv::Mat &_img, const cv::Mat &_mask) {
	_fdetector->detect(_img, _kps, _mask);
	anms(1500);
	_fdetector->compute(_img, _kps, desc);
}

void Frame::anms(const int n) {
	if (_kps.size() < n) 
		return;
	sort(_kps.begin(), _kps.end(), [&](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {return lhs.response > rhs.response;});
	std::vector<cv::KeyPoint> anms_pts;
	std::vector<double> rad_i, rad_i_sorted;
	rad_i.resize(_kps.size());
	rad_i_sorted.resize(_kps.size());

	const float c_robust = 1.1111;

	for (int i=0; i<_kps.size(); i++) {
		const float response = _kps[i].response * c_robust;
		double radius = std::numeric_limits<double>::max();
		for (int j=0; j<i && _kps[j].response > response; j++) {
			radius = std::min(radius, cv::norm(_kps[i].pt - _kps[j].pt));
		}
		rad_i[i] = radius;
		rad_i_sorted[i] = radius;
	}
	sort(rad_i_sorted.begin(), rad_i_sorted.end(), [&](const double &lhs, const double &rhs) {return lhs > rhs;});
	const double final_radius = rad_i_sorted[n-1];
	for (int i=0; i<rad_i.size(); i++) {
		if (rad_i[i] >= final_radius) {
			anms_pts.push_back(_kps[i]);
		}
	}
	_kps.swap(anms_pts);
}


void Frame::normalize_kps(const Mat3d &_K_inv) {
	kpus_mat = MatXd(_kps.size(), 2);
	kps_mat = MatXd(_kps.size(),2);
	kps_mat.setZero();
	MatXd homo_kps = MatXd(_kps.size(), 3);
	for (int i=0; i<_kps.size(); i++){
		Vec2d temp0;
		Vec3d temp;
		temp0 << _kps[i].pt.x, _kps[i].pt.y;
		temp << _kps[i].pt.x, _kps[i].pt.y, 1.0;
		kpus_mat.row(i) = temp0;
		homo_kps.row(i) = temp;
	}
	kps_mat = (_K_inv * homo_kps.transpose()).transpose().block(0,0,_kps.size(),2);
}

bool Frame::isEmpty() {
	if (pts_idx_i.size()==0) {
		return true;
	}
	return false;
}

bool Frame::getPointAtIdx(int _idx, PointPtr &_pt_res) {
	if (!pts_idx_i.count(_idx))
		return false;
	int i = pts_idx_i[_idx];
	wPointPtr _wpt = pts[i];
	PointPtr _pt = _wpt.lock();
	if (_pt)
		_pt_res = _pt;
	return true;
}

void Frame::match_frames(const FramePtr &f, std::vector<int>& _idx1, std::vector<int>& _idx2, std::vector<std::pair<Vec2d, Vec2d>>& _ret, Log *_log){
	std::vector<std::vector<cv::DMatch>> matches;
	_fmatcher->knnMatch(desc, f->desc, matches, 2, cv::noArray(), false);
	_log->mark_matchesInfo(matches.size());

	std::unordered_set<int> idx1s, idx2s;
	// Lowe's ratio test
	for (const std::vector<cv::DMatch>& m : matches){
		if (m[0].distance < ratio_threshold*m[1].distance) {
			Vec2d pt1 = kps_mat.row(m[0].queryIdx);
			Vec2d pt2 = f->kps_mat.row(m[0].trainIdx);
			if (m[0].distance > 31)
				continue;
			if (idx1s.count(m[0].queryIdx))
				continue;
			if (idx2s.count(m[0].trainIdx))
				continue;
			_idx1.push_back(m[0].queryIdx);
			_idx2.push_back(m[0].trainIdx);
			idx1s.insert(m[0].queryIdx);
			idx2s.insert(m[0].trainIdx);
			_ret.push_back(std::make_pair(pt1, pt2));
		}
	}
	assert(_idx1.size() == idx1s.size());
	assert(_idx2.size() == idx2s.size());
	_log->mark_matchesInfo(_idx1.size());
}

MatXd Frame::triangulate(const FramePtr &f, const std::vector<int>& idx1, const std::vector<int>& idx2){
	MatXd ret(idx1.size(), 4);
	ret.setZero();
	for (int i = 0; i<idx1.size(); i++){
		Mat4d A, Vt, pose2;
		A.setZero();
		pose2 = f->getPose();
		A.block(0,0,1,4) = kps_mat.row(idx1[i])(0) * _pose.block(2,0,1,4) - _pose.block(0,0,1,4);
		A.block(1,0,1,4) = kps_mat.row(idx1[i])(1) * _pose.block(2,0,1,4) - _pose.block(1,0,1,4);
		A.block(2,0,1,4) = f->kps_mat.row(idx2[i])(0) * pose2.block(2,0,1,4) - pose2.block(0,0,1,4);
		A.block(3,0,1,4) = f->kps_mat.row(idx2[i])(1) * pose2.block(2,0,1,4) - pose2.block(1,0,1,4);
		Eigen::JacobiSVD<MatXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Vt = svd.matrixV().transpose();
		ret.block(i,0,1,4) = Vt.block(3,0,1,4);
	} 
	return ret;
}

FramePair::FramePair(const cv::Mat &_imgL, const cv::Mat &_imgR, const Map &_map) {
	left = std::make_shared<Frame>(_imgL, _map.K_inv, _map);
	right = std::make_shared<Frame>(_imgR, _map.K_inv, _map);
	lr = {left, right};
	_sgbm = _map.sgbm;
	
}

void FramePair::setPoseIdentity() {
	left->setPoseIdentity();
}

void FramePair::setRightPose() {
	right->setPose(poseR* left->getPose());
	_right_empty = false;
}

void FramePair::match_framePair(Log *_log){
	std::vector<std::pair<Vec2d, Vec2d>> ret;
	left->match_frames(right, idxL, idxR, ret, _log);
}

void FramePair::disparity(const cv::Mat &_imgL, const cv::Mat &_imgR, Map &_map, const cv::Mat &_imgcL, const cv::Mat &_imgcR, Log *_log) {
	cv::Mat disparity_sgbm;
	_sgbm->compute(_imgL, _imgR, disparity_sgbm);
	disparity_sgbm.convertTo(_disparity, CV_32F, 1.0 / 16.0f);

	int stereo_disparity_pts_count = 0;
	for (const wFramePtr &_wf : lr) {
		FramePtr _f = _wf.lock();
		if (!_f)
			continue;
		for (size_t idx=0; idx<=_f->kps().size()-1; idx++) {
			cv::KeyPoint _kp = _f->kps()[idx];
			Vec3d rel_depth;
			if (!stereoDepth(_kp, _map.K, rel_depth))
				continue;
			Vec4d rel_depth_homo;
			rel_depth_homo << rel_depth, 1.0;
			if (rel_depth(2) > 5 && rel_depth(2) < 500) {
				Vec3d kp_3d = _f->getPose().inverse().block(0,0,3,4)*rel_depth_homo;
				const cv::Mat* img_ptr;
				if (_f->id()%2==0)
				 	img_ptr = &_imgcL;
					else {
					img_ptr = &_imgcR;
				}
				cv::Vec<uchar,3> p_c = img_ptr->at<cv::Vec<uchar,3>>(_kp.pt.y, _kp.pt.x);
				cv::Vec3f pixelColor = p_c;
				pixelColor/=255.0f;
				PointPtr pt = std::make_shared<Point>(kp_3d, pixelColor);
				_map.add_point(pt);
				pt->add_observation(idx, _f);
				stereo_disparity_pts_count++;
			}
		}
	}
	_log->mark_ptsInfo(stereo_disparity_pts_count);
}

bool FramePair::stereoDepth(const cv::KeyPoint &_kp, const Mat3d &_K, Vec3d &_res) {
	double _val, _fxy, cx, _cy, _b, _x, _y, _depth;
	val = _disparity.at<float>(_kp.pt.y, _kp.pt.x);
	if (val < 0)
		assert(false);
		return false;
 	_x = (_kp.pt.x - _K(6)) / K(0);
  _y = (_kp.pt.y - _K(7)) / K(0);
  _depth = K(0) / val;
	if (_depth < 0) {
		std::cerr<<"depth : "<<_depth<<"\n";
		assert(false);
		return false;
	}
	_res = Vec3d(_x*_depth, _y*_depth, _depth);
	return true;
}

