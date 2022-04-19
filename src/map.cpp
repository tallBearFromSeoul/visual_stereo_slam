#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"
#include "log.hpp"
#include <numeric>

void Map::add_point(const PointPtr &_pt) {
	_pt->set_id(max_point++);
	map_points.push_back(_pt);
	if (map_points_ids.count(_pt->id())) {
		assert(false);
	}
	map_points_ids.insert(_pt->id());
}

void Map::add_frame(const FramePtr &_f) {
	_f->set_id(max_frame++);
	map_frames.push_back(_f);
}

void Map::add_framePair(const FramePairPtr &_F) {
	_F->set_id(max_framePair++);
	add_frame(_F->left);
	add_frame(_F->right);
	map_framePairs.push_back(_F);
}

void Map::add_keyFramePair(const FramePairPtr &_F) {
	_F->left->set_kf();
	_F->right->set_kf();
	map_keyFramePairs.push_back(_F);
	map_keyFrames.push_back(_F->left);
	map_keyFrames.push_back(_F->right);
	add_keyFramePoints(_F->left->pts);
	add_keyFramePoints(_F->right->pts);
}

void Map::add_keyFramePoints(const vec_wPointPtr &_pts) {
	for (const wPointPtr &_wpt : _pts) {
		PointPtr _pt = _wpt.lock();
		if (!_pt)
			continue;
		map_keyFramePoints.push_back(_pt);
	}
}

void Map::cull(Log *_log) {
	if (map_points.empty())
		return;
	// pruning points
	double culling_threshold = 0.05;
	int culled_pts_count = 0;

	for (vec_PointPtr::iterator it=map_points.begin(); it!=map_points.end(); ) {
		PointPtr _pt = *it;
		// computing reprojection error
		std::vector<double> errors;
		for (const wFramePtr &_wf : _pt->frames) {
			FramePtr _f = _wf.lock();
			if (!_f)
				continue;
			Vec2d kp = _f->kps_mat.row(_pt->f_id_idx[_f->id()]);
			Eigen::Vector3d proj = _f->getPose().block(0,0,3,4) * _pt->homogeneous();
			proj /= proj(2);
			double proj_error = (proj.block(0,0,2,1)-kp).norm();
			errors.push_back(proj_error);
		}
		// culling
		if (errors.empty()) {
			++it;
			continue;
		}
		double mean_errors = accumulate(errors.begin(), errors.end(), 0.0)/double(errors.size());
		if (mean_errors > culling_threshold) {
			culled_pts_count++;
			_pt->delete_point();
			map_points.erase(it);
		} else {
			++it;
		}
	}
	_log->mark_cullInfo(culled_pts_count);

	int culled_frames_count = 0;
	for (vec_FramePtr::iterator it=map_frames.begin(); it!=map_frames.end(); ) {
		FramePtr _f = *it;
		if (_f->id()%2==0 && (_f->isEmpty() || _f->id()+VOLD<max_frame)) {
			culled_frames_count++;
			map_frames.erase(it);
			if (STEREO) {
				map_frames.erase(it);
			}
		} else {
			++it;
		}
	}
	_log->mark_cullInfo(culled_frames_count);
}

void Map::refresh(Log *_log) {
	int refreshed_fps_count = 0;
	for (vec_FramePairPtr::iterator it=map_framePairs.begin(); it!=map_framePairs.end(); ) {
		FramePairPtr _F = *it;
		if (_F->left.use_count() == 1 || _F->right.use_count() == 1) {
			map_framePairs.erase(it);
			refreshed_fps_count++;
		} else {
			_F->setRightPose();
			++it;
		}
	}
	if (refreshed_fps_count > 0) {
		assert(false);
	}	
}

void Map::refresh_map(Log *_log) {
	for (const FramePairPtr &_F : map_framePairs) {
		_F->setRightPose();
	}
}

double Map::optimize_map(int local_window, bool global, bool KF, bool fix_points, int rounds, bool _verbose, Log *_log) {
	double error;
	if (KF) {
		error = optimize(map_keyFrames, map_keyFramePoints, local_window, global, fix_points, rounds, _verbose);
	} else {
		error = optimize(map_frames, map_points, local_window, global, fix_points, rounds, _verbose);
	}
	if (global) {
		//refresh(_log);
		//cull(_log);
		refresh_map(_log);
	}
	return error;
}

double Map::optimize(const vec_FramePtr &frames, const vec_PointPtr &points, int local_window, bool global, bool fix_points, int rounds, bool _verbose) {
		bool verbose = _verbose;
		vec_FramePtr local_frames;
		if (local_window == -1)
			local_window = static_cast<int>(frames.size()-1);
			local_window = local_window > SLIDING_WINDOW ? SLIDING_WINDOW : local_window;
		if (global && STEREO) {
			for (size_t i=frames.size()-local_window; i<=frames.size()-1; i++) {
				if (frames[i]->id()%2==0) {
					local_frames.push_back(frames[i]);
				}
			}
		} else {
			for (size_t i=frames.size()-local_window; i<=frames.size()-1; i++) {
				local_frames.push_back(frames[i]);
			}
		}
		std::unordered_set<FramePtr> local_frames_set(local_frames.begin(), local_frames.end());
		// create g2o opt
		g2o::SparseOptimizer opt;
		opt.setVerbose(verbose);
		const std::string solverName = "lm_fix6_3_csparse";
		g2o::OptimizationAlgorithmProperty solverProperty;
		opt.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty));
		
		//g2o::OptimizationAlgorithm* opt_algo = g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty);
		//opt.setAlgorithm(opt_algo);
		// add normalized camera
		double focal_length = 1.0;
		Vec2d principal_point(0.0,0.0);
		g2o::CameraParameters* cam_params = new g2o::CameraParameters(focal_length, principal_point, 0.);
		cam_params->setId(0);
		if (!opt.addParameter(cam_params)) {
			assert(false);
		}

		std::unordered_map<FramePtr, g2o::VertexSE3Expmap*> graph_frames;
		if (fix_points) {
			for (const FramePtr &_f : local_frames) {
				g2o::SE3Quat se3(_f->getPose().block(0,0,3,3), _f->getPose().block(0,3,3,1));
				g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
				v_se3->setEstimate(se3);
				v_se3->setId(_f->id()*2);
				v_se3->setFixed(_f->id() <=1);
				opt.addVertex(v_se3);
				g2o::SE3Quat est = v_se3->estimate();
				assert(_f->getPose().isApprox(est.to_homogeneous_matrix(), 1e-5));
				graph_frames[_f] = v_se3;
			}
		} else {
			for (const FramePtr &_f : frames) {
				g2o::SE3Quat se3(_f->getPose().block(0,0,3,3), _f->getPose().block(0,3,3,1));
				g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
				v_se3->setEstimate(se3);
				v_se3->setId(_f->id()*2);
				if (_f->id() <= 1 || !local_frames_set.count(_f)) {
					v_se3->setFixed(true);
				}
				opt.addVertex(v_se3);
				g2o::SE3Quat est = v_se3->estimate();
				assert(_f->getPose().isApprox(est.to_homogeneous_matrix(), 1e-5));
				graph_frames[_f] = v_se3;
			}	
		}
		// add points to frames
		std::unordered_map<PointPtr, g2o::VertexPointXYZ*> graph_points;
		for (const PointPtr &_pt : points) {
			bool present = false;
			for (const wFramePtr &_wf : _pt->frames) {
				FramePtr _f = _wf.lock();
				if (!_f)
					continue;
				if (local_frames_set.count(_f)) {
					present = true;
				}
			}
			if (!present) {
				continue;
			}
			g2o::VertexPointXYZ* v_pt = new g2o::VertexPointXYZ();
			v_pt->setId(_pt->id()*2+1);
			v_pt->setMarginalized(true);
			v_pt->setEstimate(_pt->xyz);
			v_pt->setFixed(fix_points);
			opt.addVertex(v_pt);
			graph_points[_pt] = v_pt;
			// add edges
			for (const wFramePtr &_wf : _pt->frames) {
				FramePtr _f = _wf.lock();
				if (!_f)
					continue;
				if (!graph_frames.count(_f)){
					continue;
				}
				g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
				e->setParameterId(0,0);

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_pt));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(graph_frames[_f]));
				Vec2d edge = (_f->kps_mat.row(_pt->f_id_idx[_f->id()]));
				e->setMeasurement(edge);
				e->information() = Eigen::Matrix2d::Identity();
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				rk->setDelta(sqrt(5.991));
				e->setRobustKernel(rk);
				opt.addEdge(e);
			}
		}
		opt.initializeOptimization();
		opt.optimize(rounds);
			
		// put frames back
		for (std::unordered_map<FramePtr, g2o::VertexSE3Expmap*>::iterator it = graph_frames.begin(); it!=graph_frames.end(); ++it){
			g2o::SE3Quat est = it->second->estimate();
			it->first->setPose(est.to_homogeneous_matrix());
		}
		// put points back
		if (!fix_points){
			for (std::unordered_map<PointPtr, g2o::VertexPointXYZ*>::iterator it = graph_points.begin(); it!=graph_points.end(); ++it){
				it->first->xyz = it->second->estimate();
			}
		}
		return opt.activeChi2();	
}
