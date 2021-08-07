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
// Originally by Matt Overby
// Modified by George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

#ifndef SRC_SOLVER_HPP_
#define SRC_SOLVER_HPP_

#include <vector>
#include <memory>

#include "Settings.hpp"
#include "System.hpp"
#include "ADMM.hpp"

namespace wrapd {

// The main solver
class Solver {
 public:
    explicit Solver(const Settings& settings);

    // Adds nodes to the Solver.
    // Returns the current total number of nodes after insert.
    // Assumes m is scaled x3 (i.e. 3 values per node).
    template <typename T>
    int add_nodes(T *x, T *m, int n_verts);

    // Pins vertex indices to the location indicated. If the points
    // vector is empty (or not the same size as inds), vertices are pinned in place.
    virtual void set_pins(const std::vector<int> &inds,
            const std::vector<math::Vec3> &points = std::vector<math::Vec3>() );

    // Returns true on success.
    virtual bool initialize(const Settings &settings_ = Settings());

    // Performs a Solver step
    virtual void solve();

    // Returns the current settings
    const Settings &settings() { return m_settings; }

    std::shared_ptr<System> m_system;

    AlgorithmData* algorithm_data() {
        return m_admm.algorithm_data();
    }

    Settings m_settings;  // copied from init

 protected:
    ADMM m_admm;
    bool initialized;  // has init been called?
};  // end class Solver

}  // namespace wrapd

#endif  // SRC_SOLVER_HPP_
