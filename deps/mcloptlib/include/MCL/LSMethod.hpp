#ifndef MCL_LSMETHOD_H
#define MCL_LSMETHOD_H

namespace mcl {
namespace optlib {

// The different line search methods currently implemented
enum class LSMethod {
	None = 0, // use step length = 1, not recommended
	MoreThuente, // TODO test this one for correctness
	Backtracking, // basic backtracking with sufficient decrease
	BacktrackingCubic, // backtracking with cubic interpolation
	WeakWolfeBisection // slow
};

}
}

#endif