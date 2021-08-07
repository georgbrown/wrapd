// The MIT License (MIT)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MCL_SPLITPROBLEM_H
#define MCL_SPLITPROBLEM_H

#if MCL_DEBUG == 1
#include <iostream>
#endif

#include <Eigen/Dense>

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class SplitProblem {
private:
	typedef Eigen::Matrix<Scalar,DIM,3> MatX3;

public:
	// Returns true if the solver has converged
	// x0 is the result of the previous iteration
	// x1 is the result at the current iteration
	// grad is the gradient at the last iteration
	virtual bool converged(const MatX3 &x0, const MatX3 &x1, const MatX3 &grad) = 0;

	// Compute just the value
	virtual Scalar value(const MatX3 &x) = 0;

	// Compute the objective value and the gradient
	virtual Scalar gradient(const MatX3 &x, MatX3 &grad) = 0;

	virtual void gradient(MatX3 &grad) = 0;
};

}
}

#endif
