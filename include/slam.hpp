#include <chrono>

using namespace std::chrono;

class Map;
class Log;
class SLAM {
	private:
	protected:
		double W_inlier_ratio = 30.0;
		double H_inlier_ratio = 10.0;
		double reproj_threshold = 0.5;
		double sbp_threshold = 2.0;
		double b_dist_threshold = 48.0;
		double keyFrame_threshold = 5.0;

		int SLIDING_WINDOW = 30;
		double W;
		double H;
		Mat3d K;

		int opt_epochs = 30;//50;
		bool verbose_map = false;
		int global_opt_freq = 5;

		bool _is_log = false;
		Log* _log;

		torch::jit::script::Module module;

	public:
		Map map;
		SLAM() {};
		SLAM(double _W, double _H, const Mat3d &_K) {W=_W; H=_H; K=_K; map=Map(_K);}; 

		void load_torch_module(const std::string &_s) {try {module=torch::jit::load(_s);module.to(at::kCUDA);} catch (const c10::Error &_e) {std::cerr<<"error loading the module\n";assert(false);}};

		void reset_log();
		void print_log(const int &, bool);
		void mark_timing(const duration<double> &);
		void mark_matchesInfo(const int &);
		void mark_ptsInfo(const int &);
		void mark_mapInfo(const int &, const int &, const int &, const int &);

		void attach_log(Log* _L) {_log = _L; _is_log = true;};
		bool is_log() {return _is_log;};
		Log* log() {return _log;};
		Vec3d relTrans(const FramePtr &_f1, const FramePtr &_f2) {
			return _f1->getTrans() - _f2->getTrans();
		};
		Vec3d relTrans(const FramePairPtr &_F1, const FramePairPtr &_F2) {
			return _F1->getLeftTrans()-_F2->getLeftTrans();
		};
		double relTransNorm(const FramePtr &_f1, const FramePtr &_f2) {
		return relTrans(_f1, _f2).norm();
		};
		double relTransNorm(const FramePairPtr &_F1, const FramePairPtr &_F2) {
			return relTrans(_F1, _F2).norm();
		};
		double relTransNorm(const FramePairPtr &_F1, const wFramePairPtr &_wF2) {
			FramePairPtr _F2 = _wF2.lock();
			if (!_F2)
				assert(false);
			return relTransNorm(_F1, _F2);
		};
		virtual void process_frame(const cv::Mat &_img);
		//virtual void process_frame_cuda(const cv::cuda::GpuMat &_img);
		
		void estimate_pose(const FramePtr &, const FramePtr &, std::vector<int> &, std::vector<int> &);
		
		void add_from_prev_frame(const FramePtr &, const FramePtr &, const std::vector<int> &, const std::vector<int> &);
		void search_by_projection(const FramePtr &);
		void search_by_triangulation(const FramePtr &, const FramePtr &, const std::vector<int> &, const std::vector<int> &, const cv::Mat &);

};


class StereoSLAM : public SLAM {
	private:
		
	public:
		StereoSLAM(double _W, double _H, const Mat3d &_K) {W=_W; H=_H; K=_K; map=Map(_K, true);}; 

		void process_frame(const cv::Mat &, const cv::Mat &);
		void process_frame_cuda(cv::cuda::GpuMat &, cv::cuda::GpuMat &);

		void disparity();
		void search_by_projection(const FramePairPtr &_F) {SLAM::search_by_projection(_F->left);SLAM::search_by_projection(_F->right);};
};


