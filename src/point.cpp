#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"

int Point::latest_f_id() {
	int latest_id = -1;
	for (const wFramePtr &_wf : frames) {
		FramePtr _f = _wf.lock();
		if (!_f)
			continue;
		latest_id = _f->id() > latest_id ? _f->id() : latest_id;
	}
	if (latest_id == -1)
		assert(false);
	return latest_id;
}

void Point::add_observation(int idx, const FramePtr &_f) {
	bool checks = !_f->pts_idx_i.count(idx) && !f_id_idx.count(_f->id()) && !frames_ids.count(_f->id());
	assert(checks);
	
	_f->pts_idx_i[idx] = int(_f->pts.size());
	_f->pts.push_back(shared_from_this());
	f_id_idx[_f->id()] = idx;
	frames.push_back(_f);
	frames_ids.insert(_f->id());
}

void Point::delete_point() {
	for (const wFramePtr &_wf : frames) {
		FramePtr _f = _wf.lock();
		if (!_f)
			continue;
		int idx = f_id_idx[_f->id()];
		int i = _f->pts_idx_i[idx];
		if (!_f->pts_idx_i.count(idx))
			assert(false);
		_f->pts_idx_i.erase(idx);
		if (!frames_ids.count(_f->id()))
			assert(false);
		frames_ids.erase(_f->id());
	}
}

Vec4d Point::homogeneous() {
	Vec4d homo_pt;
	homo_pt	<< xyz, 1.0;
	return homo_pt;
}

double Point::hamming_distance(const cv::Mat &a, const cv::Mat &b) {
	double dist_ham = cv::norm(a, b, cv::NORM_HAMMING);
	return dist_ham;
}

double Point::orb_distance(const cv::Mat &desc) {
	double min_dist_ham = std::numeric_limits<double>::max();
	for (const wFramePtr &_wf : frames) {
		FramePtr _f = _wf.lock();
		if (!_f)
			continue;
		cv::Mat desc2 = _f->desc.row(f_id_idx[_f->id()]);
		double dist_ham = hamming_distance(desc, desc2);
		min_dist_ham = dist_ham < min_dist_ham ? dist_ham : min_dist_ham;
	}
	return min_dist_ham;
}

