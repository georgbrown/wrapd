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

#ifndef SRC_ALGORITHMDATA_HPP_
#define SRC_ALGORITHMDATA_HPP_

#include <memory>
#include <vector>

#include "Math.hpp"
#include "Settings.hpp"
#include "System.hpp"
#include "RuntimeData.hpp"
#include "MCL/MicroTimer.hpp"

namespace wrapd {

class AlgorithmData {
 public:
    AlgorithmData(const Settings& settings, std::shared_ptr<System> system);

    void log_initial_data();

    const std::vector<double>& per_iter_objectives() const {
        return m_objectives;
    }

    const std::vector<int>& per_iter_reweighted() const {
        return m_reweighted;
    }

    const std::vector<double>& per_iter_state_x_errors() const {
        return m_state_x_errors;
    }

    double current_normalized_x_error() const {
        return (m_state_x_errors[m_iter]);
    }

    double current_normalized_objective_residual() const {
        return m_norm_obj_residuals[m_iter-1];
    }

    bool no_recent_flips() const {
        return (m_flips_count[m_iter] == 0 && m_flips_count[m_iter-1] == 0);
    }

    const std::vector<double>& per_iter_accumulated_time_s() const {
        return m_accumulated_time_s;
    }

    const std::vector<int>& per_iter_inner_iters() const {
        return m_per_iter_inner_iters;
    }

    void get_history_x(std::vector<math::MatX3>& history_x) {
        history_x = m_history_x;
    }

    void update_per_iter_reweighted(bool reweighted) {
        if (reweighted) {
            m_reweighted.push_back(1);
        } else {
            m_reweighted.push_back(0);
        }
    }
    void update_per_iter_delta_auglag(double delta_ls, double delta_gs) {
        m_per_iter_ls_delta_auglag.push_back(std::fabs(delta_ls));
        m_per_iter_gs_delta_auglag.push_back(std::fabs(delta_gs));
    }
    void update_per_iter_data();
    void update_per_iter_runtime();

    double primal_residual() const {
        return m_system->primal_residual();
    }

    void print_initial_data() const;
    void print_curr_iter_data() const;
    void print_final_iter_data() const;

    inline int num_fix_verts() const { return m_S_fix.cols(); }
    inline int num_free_verts() const { return m_S_free.cols(); }

    RuntimeData m_runtime;  // reset each iteration
    double m_alphasq_dtsq;
    double m_init_state_x_error;
    double m_init_objective;
    int m_iter;

    std::vector<math::MatX3> m_history_x;
    std::vector<double> m_norm_obj_residuals;
    std::vector<int> m_per_iter_inner_iters;
    std::vector<double> m_per_iter_ls_delta_auglag;
    std::vector<double> m_per_iter_gs_delta_auglag;

    math::MatX3 m_prev_x_free;
    math::MatX3 m_soln_x_free;

    math::MatX3 m_curr_x_free;
    math::MatX3 m_curr_x_fix;
    math::MatX3 m_init_x_fix;

    math::SpMat m_S_fix;
    math::SpMat m_S_free;
    math::VecXi m_positive_pin;

    double m_init_auglag_obj;  // Only used by Polar right now
    double m_last_localstep_delta_auglag_obj;  // ""

 protected:

    void reset_fix_free_S_matrix();

    const Settings& m_settings;
    std::shared_ptr<System> m_system;
    Settings::RotAwareness m_rot_awareness;
    bool m_soln_provided;

    std::vector<double> m_objectives;
    std::vector<double> m_state_x_errors;
    std::vector<int> m_reweighted;
    std::vector<double> m_accumulated_time_s;
    std::vector<int> m_flips_count;


};  // end of class AlgorithmData

}  // namespace wrapd

#endif  // SRC_DATA_HPP_
