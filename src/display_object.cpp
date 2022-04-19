
void Display::DrawVehicle(const BBox &bbox) {
	double x = 3;
	double y = 2;
	double z = 3;
	double x2 = 2;

	glBegin(GL_LINES);
	glVertex3f(x,0,0);
	glVertex3f(x,0,z);
	glVertex3f(x,0,z);
	glVertex3f(x,y,z);
	glVertex3f(x,y,z);
	glVertex3f(x,y,0);
	glVertex3f(x,y,0);
	glVertex3f(x,0,0);

	glVertex3f(x2,0,0);
	glVertex3f(x2,0,z);
	glVertex3f(x2,0,z);
	glVertex3f(x2,y,z);
	glVertex3f(x2,y,z);
	glVertex3f(x2,y,0);
	glVertex3f(x2,y,0);
	glVertex3f(x2,0,0);

	glVertex3f(x,0,0);
	glVertex3f(x2,0,0);
	glVertex3f(x2,0,0);
	glVertex3f(x2,y,0);
	glVertex3f(x2,y,0);
	glVertex3f(x,y,0);
	glVertex3f(x,y,0);
	glVertex3f(x,0,0);
	
	glVertex3f(x,0,z);
	glVertex3f(x2,0,z);
	glVertex3f(x2,0,z);
	glVertex3f(x2,y,z);
	glVertex3f(x2,y,z);
	glVertex3f(x,y,z);
	glVertex3f(x,y,z);
	glVertex3f(x,0,z);
	glEnd();
}

void Display::DrawVehicles(const vector<BBox> &bboxes, const Mat4d &pose) {
	double a[16];	
	double *p = &a[0];
	Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(p, 4, 4) = pose;
	
	glPushMatrix();
	glMultTransposeMatrixd(p);

	for (const BBox &bbox : bboxes) {
		if (bbox.size() != 4)
			assert(false);
		double center_x = bbox[0];
		double center_y = bbox[1];
		double width = bbox[2];
		double height = bbox[3];

		glColor3f(0.5,0.5,1.0);
		DrawVehicle(bbox);
	}
	glPopMatrix();
}


