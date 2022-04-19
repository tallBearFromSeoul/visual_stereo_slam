#include <vector>
#include <chrono>
#include <iostream>

using namespace std::chrono;

enum Prints {
	READ_IMAGES,
	MATCH_FRAMES,
	RANSAC,
	POSE_FROM_E,
	CONVERT2GRAY,
	ESTIMATING_POSE,
	ADD_FROM_PREV_FRAME,
	LOCAL_OPTIMIZATION,
	SEARCH_BY_TRIANGULATION,
	STEREO_SGM,
	PROCESS_FRAME,
	RENDERING,
	END_OF_FRAME
};

class Log {
	private:
		std::vector<duration<double>> _durations;
		std::vector<int> _matchesInfo;
		std::vector<int> _ptsInfo;
		std::vector<int> _mapInfo;
		std::vector<int> _cullInfo;
		std::vector<duration<double>> _durations_1;
	
	public:
		std::vector<duration<double>> durations() {return _durations;};
		std::vector<duration<double>> durations_1() {return _durations_1;};
		std::vector<int> matchesInfo() {return _matchesInfo;};
		std::vector<int> ptsInfo() {return _ptsInfo;};
		std::vector<int> mapInfo() {return _mapInfo;};
		std::vector<int> cullInfo() {return _cullInfo;};

		void mark_timing(const duration<double> &_diff) {_durations.push_back(_diff);};
		void mark_timing_1(const duration<double> &_diff) {_durations_1.push_back(_diff);};
		void mark_matchesInfo(const int &_I) {_matchesInfo.push_back(_I);};
		void mark_ptsInfo(const int &_I) {_ptsInfo.push_back(_I);};
		void mark_mapInfo(const int &_I) {_mapInfo.push_back(_I);};
		void mark_cullInfo(const int &_I) {_cullInfo.push_back(_I);};

		void reset() {_durations.clear(); _durations_1.clear(); _matchesInfo.clear(); _ptsInfo.clear(); _mapInfo.clear();};
		void print_log(const int &, bool);
		void print_info() {print_matchesInfo();print_ptsInfo();print_mapInfo();print_cullInfo();};
		void print_matchesInfo();
		void print_ptsInfo();
		void print_mapInfo();
		void print_timings(const int &);
		void print_cullInfo();
		Log() {};
};

