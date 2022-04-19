#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"
#include "display.hpp"
#include <iostream>

Display::Display(const double &_w, const double &_h) {
	const std::string window_name = "vSlam_Map";
	w = _w;
	h = _h;
	setup();
}

void Display::setup() {
	pangolin::CreateWindowAndBind(window_name, w, h);
	glEnable(GL_DEPTH_TEST);
	pangolin::GetBoundWindow()->RemoveCurrent();
	
	pangolin::BindToContext(window_name);
	glEnable(GL_DEPTH_TEST);
	s_cam = pangolin::OpenGlRenderState(
			pangolin::ProjectionMatrix(w, h, p0, p0, w/2, h/2, 0.1, 1000),
			pangolin::ModelViewLookAt(-x0, -y0, -z0,
						  0, 0, 1,
						  0, -1, 0));
}

void Display::DrawPoints(const vec_PointPtr &_vertices) {
	for (const PointPtr &_v : _vertices) {
		if (_v->isColor()) {
			glColor3f(_v->color()[2], _v->color()[1], _v->color()[0]);
		} else { 
			glColor3f(255,0,0);
		}
		glBegin(GL_POINTS);
		glVertex3f(float(_v->xyz(0)), float(_v->xyz(1)), float(_v->xyz(2)));
		glEnd();
	}
}

void Display::DrawCamera(const Mat4d &pose, bool draw_ground) {
	float w = 0.5;
	float h = 0.3;
	float z = 0.5;

	double a[16];	
	double *p = &a[0];
	Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(p, 4, 4) = pose;
	
	glPushMatrix();
	glMultTransposeMatrixd(p);

	glBegin(GL_LINES);
	glVertex3f(0,0,0);
	glVertex3f(w,h,z);
	glVertex3f(0,0,0);
	glVertex3f(w,-h,z);
	glVertex3f(0,0,0);
	glVertex3f(-w,-h,z);
	glVertex3f(0,0,0);
	glVertex3f(-w,h,z);

	glVertex3f(w,h,z);
	glVertex3f(w,-h,z);

	glVertex3f(-w,h,z);
	glVertex3f(-w,-h,z);

	glVertex3f(-w,h,z);
	glVertex3f(w,h,z);

	glVertex3f(-w,-h,z);
	glVertex3f(w,-h,z);
	glEnd();

	if (false) {
	//if (draw_ground) {
		glColor3f(0,1.0,1.0);
		for (float i=1.0f; i<=3.0f; i+=0.5) {
			glBegin(GL_LINES);
			glVertex3f(i/2,0,i);
			glVertex3f(-i/2,0,i);

			glVertex3f(-i/2,0,i);
			glVertex3f(-i/2,0,-i);
			
			glVertex3f(-i/2,0,-i);
			glVertex3f(i/2,0,-i);

			glVertex3f(i/2,0,-i);
			glVertex3f(i/2,0,i);
			glEnd();
		}
	}
	glPopMatrix();
}

void Display::DrawCameras(const vec_FramePtr &_frames) {
	float w = 1.0;
	float h = 0.75;
	float z = 0.6;
	
	glPushMatrix();
	int i=0;
	int i_max = static_cast<int>(_frames.size()-1);
	for (const FramePtr &_f : _frames) {
		if (i%2==0) {
			if (_f->is_kf()) {// || (i==i_max-1 || i==i_max)) {
				glColor3f(1.0,0.0,0.0);
			} else {
				glColor3f(0.4,0.0,0.0);
			}
		} else {
			if (_f->is_kf()) {// || (i==i_max-1 || i==i_max)) {
				glColor3f(0.0,1.0,0.0);
			} else {
				glColor3f(0.0,0.4,0.0);
			}
		}
		DrawCamera(_f->getPose().inverse(), false);
		//DrawObstacles(_f->bboxes());
		i++;
	}
	glPopMatrix();
}

void Display::DrawObstacles(const std::vector<cv::Rect_<float>> &) {
	
}

void Display::draw_map(const Map& map, DrawType _dt) {
	vec_FramePtr _vec_frames;
	vec_PointPtr _vec_points;
	switch(_dt) {
		case ALL:
			run(std::make_pair(map.map_frames, map.map_points));
			break;
		case KF_ONLY:
			for (const wFramePtr &_wf : map.map_keyFrames) {
				FramePtr _f = _wf.lock();
				if (!_f)
					continue;
				_vec_frames.push_back(_f);
			}			
			_vec_frames.push_back(map.map_frames[map.map_frames.size()-2]);
			_vec_frames.push_back(map.map_frames.back());
			for (const wPointPtr &_wpt : map.map_keyFramePoints) {
				PointPtr _pt = _wpt.lock();
				if (!_pt)
					continue;
				_vec_points.push_back(_pt);
			}
			run(std::make_pair(_vec_frames, _vec_points));
			
			//run(std::make_pair(map.map_keyFrames, map.map_keyFramePoints));
			break;
		case HYBRID:
			for (const wFramePtr &_wf : map.map_keyFrames) {
				FramePtr _f = _wf.lock();
				if (!_f)
					continue;
				_vec_frames.push_back(_f);
			}
			_vec_frames.push_back(map.map_frames[map.map_frames.size()-2]);
			_vec_frames.push_back(map.map_frames.back());

			run(std::make_pair(_vec_frames, map.map_points));
			break;
	}
}

void Display::run(const std::pair<vec_FramePtr, vec_PointPtr> &buf) {
	double pos_x0 = -x0;
	double pos_y0 = -y0;
	double pos_z0 = -z0;

	double pos_x = buf.first.back()->getPose().inverse()(12);
	double pos_y = buf.first.back()->getPose().inverse()(13);
	double pos_z = buf.first.back()->getPose().inverse()(14);
	if (buf.first.size() > 2) {
		pos_x0 += buf.first[buf.first.size()-3]->getPose().inverse()(12);
		pos_y0 += buf.first[buf.first.size()-3]->getPose().inverse()(13);
		pos_z0 += buf.first[buf.first.size()-3]->getPose().inverse()(14);
	}
	s_cam = pangolin::OpenGlRenderState(
			pangolin::ProjectionMatrix(w, h, p0, p0, w/2, h/2, 0.2, 10000),
			pangolin::ModelViewLookAt(
																pos_x0, pos_y0, pos_z0,
																pos_x, pos_y, pos_z,
						  									0, -1, 0));
	pangolin::Handler3D handler(s_cam);
	pangolin::View& d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, w/h).SetHandler(&handler);
	if (buf.first.empty() || buf.second.empty())
		return;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	d_cam.Activate(s_cam);

	glColor3f(1.0, 1.0, 0.0);
	DrawCameras(buf.first);
	
	DrawPoints(buf.second);

	//vector<BBox> a(1,BBox(4,0));
	//DrawVehicles(a, buf.first.back());

	pangolin::FinishFrame();
}
