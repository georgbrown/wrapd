// Copyright (c) 2021 George E. Brown and Rahul Narain
//
// WRAPD uses the MIT License (https://opensource.org/licenses/MIT)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// By George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

#ifndef SRC_RUNTIMEDATA_HPP_
#define SRC_RUNTIMEDATA_HPP_

#include "Settings.hpp"

namespace wrapd {

struct RuntimeData {
    double setup_solve_ms;
    double local_ms;  // total ms for local solver
    double global_ms;  // total ms for global solver
    double refactor_ms;
    int inner_iters;  // total global step iterations
    double finish_solve_ms;
    RuntimeData() :
            setup_solve_ms(0),
            local_ms(0),
            global_ms(0),
            refactor_ms(0),
            inner_iters(0),
            finish_solve_ms(0) {}
};

}  // namespace wrapd

#endif  // SRC_RUNTIMEDATA_HPP_
