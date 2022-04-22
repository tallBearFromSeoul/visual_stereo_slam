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

bool FramePair::stereo_disparity(const cv::cuda::GpuMat &_imgL, const cv::cuda::GpuMat &_imgR, Map &_map, const cv::Mat &_imgcL, const cv::Mat &_imgcR, Log *_log) {
	disparity(_imgL, _imgR, _map, _imgcL, _imgcR, _log);
	if (obstacles_depth_est(_map.K)) {
		return true;
	} else {
		return false;
	}
}

template<typename D>
void FramePair::depth_estimate(Obs *_obs_ptr, const FramePtr &_f, const Eigen::Matrix<D,3,3> &_K, const Eigen::Matrix<D,4,4> &_pose) {
	float c_x, x, c_y, y, disp_val, depth;
	c_x = static_cast<float>(_obs_ptr->center()(0));
	c_y = static_cast<float>(_obs_ptr->center()(1));
	disp_val = _disparity.at<float>(static_cast<int>(std::roundl(c_y)), static_cast<int>(std::roundl(c_x)));
	x = (c_x - static_cast<float>(_K(6))) / static_cast<float>(_K(0));
	y = (c_y - static_cast<float>(_K(7))) / static_cast<float>(_K(0));
	depth = static_cast<float>(_K(0)) / disp_val;
	if (depth < 0) {
		std::cerr<<"Detect_obstacles assertion failure\n";
		assert(false);
	}
	Vec4f rel_depth_homo;
	rel_depth_homo << x*depth, y*depth, depth, 1.0;
	Mat4f _posef = _pose.template cast<float>();
	Vec3f depth3d = _posef.inverse().block(0,0,3,4)*rel_depth_homo;
	_obs_ptr->setDepth(depth3d);
	_f->add_m_obs(_obs_ptr);
}

bool FramePair::obstacles_depth_est(const Mat3d &_K) {
	if (!_matches.size())
		return false;
	for (size_t i=0; i<_matches.size(); i++) {
		std::vector<float> match = _matches[i].top();
		int idxL = static_cast<int>(std::roundl(match[1]));
		int idxR = static_cast<int>(std::roundl(match[2]));
		Obs *obsL_ptr, *obsR_ptr;
		obsL_ptr = left->obstacles(idxL);
		obsR_ptr = right->obstacles(idxR);
		depth_estimate(obsL_ptr, left, _K, left->getPose());
		depth_estimate(obsR_ptr, right, _K, right->getPose());
		_corr_m_obs.push_back(0.5f*(obsL_ptr->depth3d() + obsR_ptr->depth3d()));
	}
	return true;
}

bool FramePair::match_obstacles() {
	for (size_t i=0; i<left->obstacles_count(); i++) {
		std::priority_queue<std::vector<float>, std::vector<std::vector<float>>, std::greater<std::vector<float>>> temp;
		for (size_t j=0; j<right->obstacles_count(); j++) {
			if (left->obstacles(i)->class_id() != 2)
				continue;
			if (left->obstacles(i)->class_id() != right->obstacles(j)->class_id())
				continue;
			float val = left->obstacles(i)->compare(right->obstacles(j));
			std::vector<float> vals = {val, i, j, left->obstacles(i)->class_id()};
			temp.push(vals);
		}
		if (temp.size())
			_matches.push_back(temp);
	}
	return true;
}

torch::Tensor FramePair::preprocess_image(const cv::Mat &_img) {
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

void FramePair::detect_obstacles(const FramePtr &_f, const std::vector<torch::jit::IValue> &_inputs, torch::jit::script::Module &_module, const Mat3d &_K, const Mat4d &_pose, const cv::_InputOutputArray &_in_mask, const cv::_InputOutputArray &_img) {
	cv::Mat _mask = _in_mask.getMat();
	bool mask_present = true, img_filled = false;
	if (_mask.empty()) {
		mask_present = false;
	}
	float w = static_cast<float>(_inputs[1].toDouble());
	float h = static_cast<float>(_inputs[2].toDouble());
	if (!_img.empty())
		img_filled = true;
	c10::Dict outputs = _module.forward(_inputs).toGenericDict();
	torch::Tensor scores = outputs.at("scores").toTensorList()[0];
	torch::Tensor pred_classes = outputs.at("pred_classes").toTensorList()[0];
	torch::Tensor pred_boxes = outputs.at("pred_boxes").toTensorList()[0];
	for (int i=0; i<scores.sizes()[0]; i++) {
		int c = pred_classes.index({i}).item<long>();
		torch::Tensor box = pred_boxes.index({i, Slice()});
		float x1, x2, y1, y2, c_x, c_y;
		x1 = box[0].item<float>();
		y1 = box[1].item<float>();
		x2 = box[2].item<float>();
		y2 = box[3].item<float>();
		/*
		x1 = x1 < 0 ? 0 : x1;
		y1 = y1 < 0 ? 0 : y1;
		x2 = x2-x1 > w ? w-x1 : x2;
		y2 = y2-y1 > h ? h-y1 : y2;
		*/
		cv::Rect_<float> bbox(x1, y1, x2-x1, y2-y1);
		c_x = x2-(x2-x1)/2;
		c_y = y2-(y2-y1)/2;
		Obs obs(c, Vec2f(bbox.width, bbox.height), Vec2f(c_x, c_y)); 
		_f->add_obstacle(obs);
		
		if (mask_present) {
			_mask(bbox) = 0;
		}
		if (img_filled) {
			cv::rectangle(_img, bbox, cv::Scalar(0,0,255), 2);
			cv::putText(_img, CLASSES[c], cv::Point(x1, y1), 2, 1.2, cv::Scalar(0,255,0));
		}
	}
	if (img_filled) {
		cv::imshow("img_clone", _img);
		cv::waitKey(500);
	}
}

FramePair::FramePair(const cv::cuda::GpuMat &_imgL_cuda, const cv::cuda::GpuMat &_imgR_cuda, const Map &_map, const cv::Mat &_imgcL, const cv::Mat &_imgcR, torch::jit::script::Module &_module) {
	left = std::make_shared<Frame>(_imgL_cuda, _imgcL, _map);
	right = std::make_shared<Frame>(_imgR_cuda, _imgcR, _map);
	cv::Mat imgcL_float, imgcR_float, maskL, maskR;
	maskL = cv::Mat::ones(imgcL_float.rows, imgcL_float.cols, 0);
	maskR = maskL.clone();
	_imgcL.convertTo(imgcL_float, CV_32FC3, 1.0f/255.0f);
	_imgcR.convertTo(imgcR_float, CV_32FC3, 1.0f/255.0f);
	torch::Tensor img_tensorL = preprocess_image(imgcL_float).to(at::kCUDA);
	torch::Tensor img_tensorR = preprocess_image(imgcR_float).to(at::kCUDA);
	std::vector<torch::Tensor> imgsL, imgsR; 
	imgsL = {img_tensorL};
	imgsR = {img_tensorR};
	std::vector<torch::jit::IValue> inL, inR;
	inL = {imgsL, float(imgcL_float.rows), float(imgcR_float.cols)};
	inR = {imgsR, float(imgcL_float.rows), float(imgcR_float.cols)};
	detect_obstacles(left, inL, _module, _map.K, left->getPose(), maskL, _imgcL);
	detect_obstacles(right, inR, _module, _map.K, right->getPose(), maskR, _imgcR);
	lr = {left, right};
	_sgm_cuda = _map.sgm_cuda;
	
	match_obstacles();
	/*
	if (!match_obstacles()) {
		std::cerr<<"match obstacle results false\n";
		assert(false);
	}
	*/
	// if the area of rectangles are similar and the obstacles count are the same if not more problems 
}

Frame::Frame(const cv::cuda::GpuMat &_img_cuda, const cv::Mat &_img, const Map &_map) {
	_is_cuda = true;
	_fdetector = _map.fdetector;
	//_fdetector_cuda = _map.fdetector_cuda;
	_fmatcher = _map.fmatcher;
	//_fmatcher_cuda = (_map.fmatcher_cuda);
	extract_features(_img);
	normalize_kps(_map.K_inv);
	//tree = kdTree();
}

FramePair::FramePair(const cv::cuda::GpuMat &_imgL_cuda, const cv::cuda::GpuMat &_imgR_cuda, const Map &_map) {
	cv::Mat imgL, imgR;
	_imgL_cuda.download(imgL);
	_imgR_cuda.download(imgR);
	left = std::make_shared<Frame>(_imgL_cuda, imgL, _map);
	right = std::make_shared<Frame>(_imgR_cuda, imgR, _map);
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


Frame::Frame(const cv::Mat &_img, const Map &_map) {
	_fdetector = _map.fdetector;
	_fmatcher = _map.fmatcher;

	extract_features(_img);
	normalize_kps(_map.K_inv);
	//tree = kdTree();
}

void Frame::extract_features(const cv::Mat &_img, const cv::_InputOutputArray &_mask) {
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
	left = std::make_shared<Frame>(_imgL, _map);
	right = std::make_shared<Frame>(_imgR, _map);
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
	double disp_val = _disparity.at<float>(_kp.pt.y, _kp.pt.x);
	if (disp_val == -1)
		return false;
	if (disp_val < 0)
		std::cerr<<"disp val : "<<_disparity.at<float>(_kp.pt.y, _kp.pt.x)<<"\n";
  double x = (_kp.pt.x - _K(6)) / _K(0);
  double y = (_kp.pt.y - _K(7)) / _K(0);
  double depth = _K(0) / (disp_val);
	if (depth < 0) 
		std::cerr<<"depth : "<<depth<<"\n";
	_res = Vec3d(x*depth, y*depth, depth);
	return true;
}

