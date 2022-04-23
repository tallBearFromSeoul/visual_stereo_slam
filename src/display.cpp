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

void Display::draw_points(const vec_PointPtr &_vertices) {
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

void Display::draw_obstacles(const FramePairPtr &_F) {
	for (const Vec3f &_obs : _F->corr_m_obs()) {
		float x,y,z;
		x=_obs(0);
		y=_obs(1);
		z=_obs(2);
		glColor3f(150.f,200.f,100.f);
		glBegin(GL_POINTS);
		glVertex3f(x, y, z);
		glEnd();
		draw_pointer(x, y, z);	
	}
}

void Display::draw_pointer(float _x, float _y, float _z) {
	float r0=4.0f;
	for (int i=0; i<4; i++) {
		draw_circle(_x, _y, _z, r0-(i*0.5f));
	}
}

void Display::draw_circle(float _x, float _y, float _z, float _r) {
	glBegin(GL_LINE_LOOP);
	for (int ii=0; ii < 32; ii++) {
		float theta = 2.0f * 3.1415926f * float(ii) / 32.f;
		float x = _r*cosf(theta);
		float z = _r*sinf(theta);
		glVertex3f(x+_x, _y, z+_z);
	}
	glEnd();
}

void Display::draw_camera(const Mat4d &pose, bool last) {
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

	if (last) {
		glColor3f(0,5.0,5.0);
		draw_pointer(0,0,0);
	}
	glPopMatrix();
}

void Display::draw_cameras(const vec_FramePtr &_frames) {
	float w = 1.0;
	float h = 0.75;
	float z = 0.6;
	
	glPushMatrix();
	int i=0;
	int i_max = static_cast<int>(_frames.size()-1);
	for (const FramePtr &_f : _frames) {
		if (i%2==0) {
			if (_f->is_kf()) {
				glColor3f(1.0,0.0,0.0);
			} else {
				glColor3f(0.4,0.0,0.0);
			}
		} else {
			if (_f->is_kf()) {
				glColor3f(0.0,1.0,0.0);
			} else {
				glColor3f(0.0,0.4,0.0);
			}
		}
		draw_camera(_f->getPose().inverse(), i==i_max);
		i++;
	}
	glPopMatrix();
}

void Display::draw_map(const Map& map, DrawType _dt) {
	vec_FramePtr _vec_frames;
	vec_PointPtr _vec_points;
	switch(_dt) {
		case ALL:
			run(std::make_pair(map.map_frames, map.map_points), {map.map_framePairs.back()});
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

void Display::run(const std::pair<vec_FramePtr, vec_PointPtr> &buf, const FramePairPtr &_buf2) {
	float pos_x0, pos_y0, pos_z0, pos_x, pos_y, pos_z;
	pos_x0 = -x0;
	pos_y0 = -y0;
	pos_z0 = -z0;
	
	Vec3f t_1 = buf.first.back()->getPose().inverse().block(0,3,3,1).cast<float>();
	pos_x = t_1(0);
	pos_y = t_1(1);
	pos_z = t_1(2);
	if (buf.first.size() > 2) {
		Vec3f t_0 = buf.first[buf.first.size()-3]->getPose().inverse().block(0,3,3,1).cast<float>();
		pos_x0 += t_0(0);
		pos_y0 += t_0(1);
		pos_z0 += t_0(2);
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
	draw_cameras(buf.first);
	
	draw_points(buf.second);
	if (_buf2!=nullptr) {
		draw_obstacles(_buf2);
	}
	pangolin::FinishFrame();
}
