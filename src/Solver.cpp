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

#include <fstream>
#include <unordered_set>
#include <unordered_map>

#include "Solver.hpp"

namespace wrapd {

Solver::Solver(const Settings &settings) : initialized(false) {
    m_settings = settings;
    m_system = std::make_shared<System>(settings);
}


void Solver::solve() {
    static int curr_frame = 0;
    
    math::MatX3 X = m_system->X();
    m_admm.solve(X);
    m_system->X(X);
    
    curr_frame += 1;
}

bool Solver::initialize(const Settings &settings) {
    m_settings = settings;

    m_system->initialize();

    m_admm.initialize(m_system, m_settings);
    m_system->set_initial_pose();


    initialized = true;

    return true;
}  // end init


void Solver::set_pins(const std::vector<int> &inds, const std::vector<math::Vec3> &points ) {
    m_system->set_pins(inds, points);
}

}  // namespace wrapd
