#include <vector>
#include <functional>
#include <opencv2/core/types.hpp>

typedef cv::KeyPoint point_t;
typedef std::vector<size_t> indexArr;
typedef std::pair<cv::KeyPoint, size_t> pointIndex;

typedef std::vector<pointIndex> pointIndexArr;
typedef std::vector<cv::KeyPoint> pointVec;

inline double dist2(const point_t &, const point_t &);

class comparer {
	public:
		size_t idx;
		explicit comparer(size_t idx_);
		inline bool compare_idx(const pointIndex&, const pointIndex&);
};

class KDNode {
public:
	typedef std::shared_ptr<KDNode> KDNodePtr;
	size_t index;
	point_t xy;
	KDNodePtr left;
	KDNodePtr right;
	bool exist;

	KDNode();
	KDNode(const point_t &, const size_t &, const KDNodePtr &, const KDNodePtr &);
	KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
	
	~KDNode();

	double coord(const size_t &);

	explicit operator bool();
	explicit operator point_t();
	explicit operator pointIndex();
};

typedef std::shared_ptr<KDNode> KDNodePtr;

class KDTree {
	KDNodePtr root;
	KDNodePtr leaf;

	KDNodePtr make_tree(const pointIndexArr::iterator &begin,
						const pointIndexArr::iterator &end,
						const size_t &length,
						const size_t &level);
	public:
		KDTree() = default;
		explicit KDTree(pointVec point_array);
		point_t nearest_point(const point_t &pt);
		pointVec neighborhood_points(const point_t &pt, const double &rad);	
		indexArr neighborhood_indices(const point_t &pt, const double &rad);
	private:
		// default caller
		KDNodePtr nearest_(const point_t &pt);
		KDNodePtr nearest_(const KDNodePtr &branch, const point_t &pt, const size_t &level, const KDNodePtr &best, const double &best_dist);
		pointIndexArr neighborhood_(const KDNodePtr &branch, const point_t &pt, const double &rad, const size_t &level);
		
};



