#include "log.hpp"

using namespace std::chrono;

void Log::print_log(const int &_n, bool _verbose=true) {
	std::cout<<"\n\n===================\nProcessing Frame #"<<_n<<"\n";
	std::cout<<"===================\n";
	if (!_verbose)
		return;
	print_info();
	print_timings(_n);
}

void Log::print_matchesInfo() {
	if (_matchesInfo.size() == 0) {
		std::cout<<"EMPTY LOG matchesInfo\n";
		return;
	}
	int matches = _matchesInfo[0];
	int lowes_matches = _matchesInfo[1];
	int inliers = _matchesInfo[2];
	std::cout<<"[# of matches\t->\t# after lowe's test\t->\t# of inliers] :\n";
	std::cout<<"["<<matches<<"\t->\t"<<lowes_matches<<"\t->\t"<<inliers<<"]\n";
}

void Log::print_mapInfo() {
	if (_mapInfo.size() == 0) {
		std::cout<<"EMPTY LOG mapInfo\n";
		return;
	}
	int nKFs = _mapInfo[0];
	int nKFpts = _mapInfo[1];
	int nFs = _mapInfo[2];
	int nFpts = _mapInfo[3];
	std::cout<<"There are "<<nKFpts<<" amount of keyFrame points and "<<nKFs<<" amount of keyFrames in map\n";
	std::cout<<"There are "<<nFpts<<" amount of points and "<<nFs<<" amount of frames in map\n";
}

void Log::print_cullInfo() {
	if (_cullInfo.size() == 0) {
		std::cout<<"EMPTY LOG cullInfo\n";
		return;
	}
	int nFpts = _cullInfo[0];
	int nFs = _cullInfo[1];
	std::cout<<"Culled "<<nFs<<" amount of frames\n";
	std::cout<<"\nCulled "<<nFpts<<" amount of points\n";
}

void Log::print_ptsInfo() {
	if (_ptsInfo.size() == 0) {
		std::cout<<"EMPTY LOG ptsInfo\n";
		return;
	}
	int stereo_disparity_pts_count = _ptsInfo[0];
	int pts_added_from_prev_obs = _ptsInfo[1];
	int pts_added_by_triangulation = _ptsInfo[2];
	std::cout<<"# of points added by SGBM is : "<<stereo_disparity_pts_count<<"\n";
	std::cout<<"# of points added from previous frame is : "<<pts_added_from_prev_obs<<"\n";
	std::cout<<"# of points added by global triangulation is : "<<pts_added_by_triangulation<<"\n";

}

void Log::print_timings(const int &_n) {
	if (_durations.size() == 0) {
		std::cout<<"EMPTY LOG durations\n";
		return;
	}
	int i=0;
	for (const std::chrono::duration<double> &diff : _durations) {
		Prints p = static_cast<Prints>(i);
		i++;
		switch(p) {
			case READ_IMAGES:
				std::cout<<"Time taken for reading image pair is "<<diff.count()<<" seconds.\n";
				break;
			case MATCH_FRAMES:
				std::cout<<"Time taken for matching frames is "<<diff.count()<<" seconds.\n";
				break;
			case RANSAC:
				std::cout<<"Time taken for ransac is "<<diff.count()<<" seconds.\n";
				break;
			case POSE_FROM_E:
				std::cout<<"Time taken for pose from essential matrix is : "<<diff.count()<<" seconds.\n";
				break;
			case CONVERT2GRAY:
			 std::cout<<"Time taken for converting images to gray is : "<<diff.count()<<"\n";
			case ESTIMATING_POSE:
				std::cout<<"Time taken for estimating pose is : "<<diff.count()<<" seconds.\n";
				break;
			case ADD_FROM_PREV_FRAME:
				std::cout<<"Time taken for adding from previous frame is : "<<diff.count()<<" seconds.\n";
				break;
			case LOCAL_OPTIMIZATION:
				std::cout<<"Time taken for local optimization is : "<<diff.count()<<" seconds.\n";
				break;
			case SEARCH_BY_TRIANGULATION:
				std::cout<<"Time taken for search by triangulation is : "<<diff.count()<<" seconds.\n";
				break;
			case STEREO_SGM:
				std::cout<<"Time taken for SGM is : "<<diff.count()<<" seconds.\n";
				break;
			case PROCESS_FRAME:
				std::cout<<"----------------\nTime taken for processing frame is "<<diff.count()<<" seconds.\n";
				break;
			case RENDERING:
				std::cout<<"Time taken for rendering is "<<diff.count()<<" seconds.\n";
				break;
			case END_OF_FRAME:
				std::cout<<"Time taken for frame #"<<_n<<" is : "<<diff.count()<<" seconds.\n";
				break;
		}
	}
}


