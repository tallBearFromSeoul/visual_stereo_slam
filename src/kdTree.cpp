#include "kdTree.hpp"

KDNode::KDNode() {
	exist = false;
}

KDNode::KDNode(const point_t &pt, const size_t &idx_, const KDNodePtr &left_, const KDNodePtr &right_) {
	xy = pt;
	index = idx_;
	left = left_;
	right = right_;
	exist = true;
}

KDNode::KDNode(const pointIndex &pi, const KDNodePtr &left_, const KDNodePtr &right_) {
	xy = pi.first;
	index = pi.second;
	left = left_;
	right = right_;
	exist = true;
}

KDNode::~KDNode() = default;

double KDNode::coord(const size_t &idx) { 
	if (idx == 0) {
		return xy.pt.x;
	} else {
		return xy.pt.y;
	}
}

KDNode::operator bool() {
	return exist;
}

KDNode::operator point_t() {
	return xy; 
}
KDNode::operator pointIndex() {
	return pointIndex(xy, index); 
}

KDNodePtr NewKDNodePtr() {
	KDNodePtr node = std::make_shared<KDNode>();
	return node;
}

comparer::comparer(size_t idx_) : idx{idx_} {};

inline bool comparer::compare_idx(const pointIndex &a, const pointIndex &b) {
	if (idx == 0){
		return (a.first.pt.x < b.first.pt.x);
	} else {
		return (a.first.pt.y < b.first.pt.y);
	}
}   

inline void sort_on_idx(const pointIndexArr::iterator &begin, const pointIndexArr::iterator &end, size_t idx) {
	comparer comp(idx);
	comp.idx = idx;

	using std::placeholders::_1;
	using std::placeholders::_2;
	nth_element(begin, begin + distance(begin, end) / 2, 
							end, bind(&comparer::compare_idx, 
							comp, _1, _2));
}


KDNodePtr KDTree::make_tree(const pointIndexArr::iterator &begin,
							const pointIndexArr::iterator &end,
							const size_t &length,
							const size_t &level) {
	if (begin == end) {
		return NewKDNodePtr();
	}
    size_t dim = 2;

	if (length > 1){
		sort_on_idx(begin, end, level);
    }
	pointIndexArr::iterator middle = begin + (length/2);
	pointIndexArr::iterator l_begin = begin;
	pointIndexArr::iterator l_end = middle;
	pointIndexArr::iterator r_begin = middle+1;
	pointIndexArr::iterator r_end = end;

	size_t l_len = length/2;
	size_t r_len = length - l_len - 1;

	KDNodePtr left;
	if (l_len > 0 && dim > 0){
		left = make_tree(l_begin, l_end, l_len, (level + 1) % dim);
	} else {
		left = leaf;
	}
	KDNodePtr right;
	if (r_len > 0 && dim > 0){
		right = make_tree(r_begin, r_end, r_len, (level + 1) % dim);
	} else {
		right = leaf;
	}
	return std::make_shared<KDNode>(*middle, left, right);
}


//pointVec : vector<cv::KeyPoint>
//pointIndex : pair<cv::KeyPoint, size_t> 
//pointIndexArr : vector<pointIndex>
KDTree::KDTree(pointVec point_array) {
    leaf = std::make_shared<KDNode>();
    // iterators
    pointIndexArr arr;
    for (size_t i = 0; i < point_array.size(); i++) {
        arr.push_back(pointIndex(point_array.at(i), i));
    }

    auto begin = arr.begin();
    auto end = arr.end();

    size_t length = arr.size();
    size_t level = 0;  // starting

    root = KDTree::make_tree(begin, end, length, level);
}

inline double dist2(const point_t& a, const point_t& b) {
	double distc = 0;
	double dx = a.pt.x - b.pt.x;
	double dy = a.pt.y - b.pt.y;
	distc = dx*dx + dy*dy;
	return distc;
}

inline double dist2(const KDNodePtr &a, const KDNodePtr &b) {
    return dist2(a->xy, b->xy);
}

KDNodePtr KDTree::nearest_(const point_t &pt) {
    size_t level = 0;
    double branch_dist = dist2(point_t(*root), pt);
    return nearest_(root, pt, level, root, branch_dist);
};

KDNodePtr KDTree::nearest_(const KDNodePtr &branch, const point_t &pt, 
							const size_t &level, const KDNodePtr &best, 
							const double &best_dist) {
    double d, dx, dx2;
    if (!bool(*branch)) {
        return NewKDNodePtr();  // basically, null
    }

    point_t branch_pt(*branch);
    size_t dim = 2;

    d = dist2(branch_pt, pt);
		if (level == 0) {
			dx = branch_pt.pt.x - pt.pt.x;
		} else {
			dx = branch_pt.pt.y - pt.pt.y;
		}
		dx2 = dx * dx;
		KDNodePtr best_l = best;
    double best_dist_l = best_dist;

    if (d < best_dist) {
        best_dist_l = d;
        best_l = branch;
    }

    size_t next_lv = (level + 1) % dim;
    KDNodePtr section;
    KDNodePtr other;
		
    // select which branch makes sense to check
    if (dx > 0) {
        section = branch->left;
        other = branch->right;
    } else {
        section = branch->right;
        other = branch->left;
    }

    // keep nearest neighbor from further down the tree
    KDNodePtr further = nearest_(section, pt, next_lv, best_l, best_dist_l);
    if (bool(further)) {
        double dl = dist2(further->xy, pt);
        if (dl < best_dist_l) {
            best_dist_l = dl;
            best_l = further;
        }
    }
    
    // only check the other branch if it makes sense to do so
    if (dx2 < best_dist_l) {
        further = nearest_(other, pt, next_lv, best_l, best_dist_l);
        if (bool(further)) {
            double dl = dist2(further->xy, pt);
            if (dl < best_dist_l) {
                best_dist_l = dl;
                best_l = further;
            }
        }
    }

    return best_l;
};

point_t KDTree::nearest_point(const point_t &pt) {
	return point_t(*nearest_(pt));
};

pointIndexArr KDTree::neighborhood_(const KDNodePtr &branch, const point_t &pt, const double &rad, const size_t &level) {
	double d, dx, dx2;

	if (!bool(*branch)) {
		// branch has no point, means it is a leaf,
		// no points to add
		return pointIndexArr();
	}

	size_t dim = 2;

	double r2 = rad * rad;

	d = dist2(point_t(*branch), pt);
	if (level == 0) {
		dx = point_t(*branch).pt.x - pt.pt.x;
	} else {
		dx = point_t(*branch).pt.y - pt.pt.y;
	}
	dx2 = dx * dx;

	pointIndexArr nbh, nbh_s, nbh_o;
	if (d <= r2) {
		nbh.push_back(pointIndex(*branch));
	}

	//
	KDNodePtr section;
	KDNodePtr other;
	if (dx > 0) {
		section = branch->left;
		other = branch->right;
	} else {
		section = branch->right;
		other = branch->left;
	}

	nbh_s = neighborhood_(section, pt, rad, (level + 1) % dim);
	nbh.insert(nbh.end(), nbh_s.begin(), nbh_s.end());
	if (dx2 < r2) {
		nbh_o = neighborhood_(other, pt, rad, (level + 1) % dim);
		nbh.insert(nbh.end(), nbh_o.begin(), nbh_o.end());
	}

	return nbh;
	};

pointVec KDTree::neighborhood_points(const point_t &pt, const double &rad) {
size_t level = 0;
pointIndexArr nbh = neighborhood_(root, pt, rad, level);
pointVec nbhp;
nbhp.resize(nbh.size());
std::transform(nbh.begin(), nbh.end(), nbhp.begin(),
			  [](pointIndex pi) { return pi.first; });
return nbhp;
}

indexArr KDTree::neighborhood_indices(const point_t &pt, const double &rad){
        size_t level = 0;
        pointIndexArr nbh = neighborhood_(root, pt, rad, level);
        indexArr nbhi;
        nbhi.resize(nbh.size());
        std::transform(nbh.begin(), nbh.end(), nbhi.begin(), [](pointIndex pi) { return pi.second; });
        return nbhi;
}

