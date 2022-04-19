#include "point.hpp"
#include "frame.hpp"
#include "map.hpp"
#include "essentialMat.hpp"
#include "display.hpp"
#include "slam.hpp"
#include "log.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>

using namespace std::chrono;

int main(int argc, const char* argv[]){
	
	if (argc < 3) {
		std::cout<<"\nPlease type :\n";
		std::cout<<"[EXE_NAME] [START_ITER] [PATH] [PATH_TO_TORCH_MODULE] [VID_ON] [STEREO] [CUDA] [PRINT_LOG]\n";
	}

	std::string PATH = "../../interlaken_00_d_images_rectified";
	std::string PATH_TO_TORCH_MODULE = "../Apr05_scripted_detr_model_504.pt";
	//std::string PATH = "../videos/test_kitti984.mp4";
	//std::string PATH = "../videos/test.mp4";
	int START = 0;
	if (argc > 1) {
		START = atof(argv[1]);
	}
	if (argc > 2) {
		PATH = argv[2];
	}
	bool VID_ON = true;
	if (argc > 3) {
		VID_ON = atof(argv[3]);
	}
	bool STEREO = true;
	if (argc > 4) {
		STEREO = atof(argv[4]);
	}
	bool CUDA = false;
	if (argc > 5) {
		CUDA = atof(argv[5]);
	}
	bool PRINT_LOG = true;
	if (argc > 6) {
		PRINT_LOG = atof(argv[6]);
	}
	double W = 1440.0;
	double H = 1080.0;
	if (!STEREO) {
		cv::VideoCapture cap(PATH);
		W = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		H = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	}
	std::cout<<"PATH : "<<PATH<<"\n";
	std::cout<<"W : "<<W<<" | H : "<<H<<"\n";

	// camera intrinsics
	Mat3d K;
	double F = 925.0;
	if (!STEREO) {
		K<<F,0,W/2.0,0,F,H/2.0,0,0,1.0;
	} else {
		K<<1164.6238115833075, 0, 713.5791168212891, 0, 1164.6238115833075, 570.9349365234375, 0, 0, 1.0;
	}
	std::cout<<"K :\n"<<K<<"\n";

	//SLAM slam(W, H, K);
	Display display(W, H);
	StereoSLAM st_slam(W, H, K);
	Log* log = new Log();
	st_slam.attach_log(log);
	time_point<high_resolution_clock> loading_time = high_resolution_clock::now();
	st_slam.load_torch_module(PATH_TO_TORCH_MODULE);
	duration<double> load_time = high_resolution_clock::now() - loading_time;
	std::cout<<"Loading time of the module is : "<<load_time.count()<<"\n";

	cv::Mat imgL, imgR;
	cv::cuda::GpuMat imgL_cuda, imgR_cuda;

	time_point<high_resolution_clock> now, t_begin;
	duration<double> diff;	
	for (int n_iter=START; n_iter<=1990; n_iter++) {
		st_slam.reset_log();

		t_begin = high_resolution_clock::now();
    std::string path_left = (boost::format("%s/left/%06d.png") % PATH % n_iter).str();
		std::string path_right = (boost::format("%s/right/%06d.png") % PATH % n_iter).str();
		imgL = cv::imread(path_left,cv::IMREAD_COLOR);
		imgR = cv::imread(path_right,cv::IMREAD_COLOR);
		diff = high_resolution_clock::now() - t_begin;
		st_slam.mark_timing(diff);
		
		now = high_resolution_clock::now();
		if (!(imgL.empty() || imgR.empty())) {
			imgL_cuda.upload(imgL);
			imgR_cuda.upload(imgR);
			st_slam.process_frame_cuda(imgL_cuda, imgR_cuda);
			//st_slam.process_frame(imgL, imgR);
			
			diff = high_resolution_clock::now() - now;
			st_slam.mark_timing(diff);
			now = high_resolution_clock::now();

			//display.draw_map(slam.map);
			//display.draw_map(st_slam.map, ALL);
			display.draw_map(st_slam.map, ALL);//HYBRID); //KF_ONLY);
			diff = high_resolution_clock::now() - now;
			st_slam.mark_timing(diff);
			
			VID_ON = false;
			if (VID_ON) {
				cv::imshow("test",imgL);
				cv::waitKey(5);
			}
			diff = high_resolution_clock::now() - t_begin;
			st_slam.mark_timing(diff);
			
			PRINT_LOG = false;
			st_slam.print_log(n_iter, PRINT_LOG);
		} else {
			break;
		}
	}
	cv::destroyAllWindows();
	return 0;
}
