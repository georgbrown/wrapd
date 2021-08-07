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

#include "AlgorithmData.hpp"
#include <iomanip>

namespace wrapd {

AlgorithmData::AlgorithmData(const Settings& settings, std::shared_ptr<System> system)
    : m_settings(settings),
      m_system(system) {

    const int num_admm_iters = m_settings.m_max_admm_iters;
    m_objectives.reserve(num_admm_iters+1);
    m_state_x_errors.reserve(num_admm_iters+1);
    m_reweighted.reserve(num_admm_iters+1);
    m_accumulated_time_s.reserve(num_admm_iters+1);
    m_flips_count.reserve(num_admm_iters+1);

    // m_history_x.resize(num_admm_iters);
    m_history_x.resize(0);
    m_norm_obj_residuals.reserve(num_admm_iters);
    m_per_iter_inner_iters.reserve(num_admm_iters);
    m_per_iter_ls_delta_auglag.reserve(num_admm_iters);
    m_per_iter_gs_delta_auglag.reserve(num_admm_iters);

    reset_fix_free_S_matrix();

}

void AlgorithmData::log_initial_data() {
    m_rot_awareness = m_settings.m_rot_awareness;
    m_soln_provided = m_settings.m_io.soln_provided();

    m_init_objective = m_system->problem_objective();
    int init_flips_count = m_system->inversion_count();

    m_objectives.push_back(m_init_objective);
    m_state_x_errors.push_back(1.0);
    m_reweighted.push_back(1.0);
    m_accumulated_time_s.push_back(0.0);
    m_flips_count.push_back(init_flips_count);

    m_init_state_x_error = (m_prev_x_free - m_soln_x_free).norm();
}

void AlgorithmData::update_per_iter_data() {
    double curr_objective = m_system->problem_objective();
    int flips_count = m_system->inversion_count();

    m_state_x_errors.push_back((m_curr_x_free - m_soln_x_free).norm() / m_init_state_x_error);
    // m_objectives.push_back(curr_objective / m_init_objective);
    m_objectives.push_back(curr_objective);
    m_flips_count.push_back(flips_count);

    double relative_error = std::fabs(m_objectives[m_iter+1] - m_objectives[m_iter]) / (std::fabs(m_objectives[m_iter+1]) + 1.0);  // the 1.0 was added to denominator for better handling of zero-energy solutions
    m_norm_obj_residuals.push_back(relative_error);

}

void AlgorithmData::update_per_iter_runtime() {
    double curr_accumulated_time_ms = m_runtime.local_ms + m_runtime.global_ms + m_runtime.refactor_ms;
    m_accumulated_time_s.push_back(curr_accumulated_time_ms / 1000.);
}

void AlgorithmData::reset_fix_free_S_matrix() {
    std::shared_ptr<ConstraintSet> constraints = m_system->constraint_set();

    const int num_all_verts = m_system->num_all_verts();
    const int pin_dof = static_cast<int>(constraints->m_pin_inds.size());
    const int free_dof = num_all_verts - pin_dof;

    m_positive_pin.setOnes(static_cast<int>(num_all_verts));

    math::SpMat S_fix(num_all_verts, pin_dof);
    math::SpMat S_free(num_all_verts, free_dof);

    Eigen::VectorXi nnz = Eigen::VectorXi::Ones(num_all_verts);  // non zeros per column
    S_fix.reserve(nnz);
    S_free.reserve(nnz);

    int count = 0;
    for (int i = 0; i < static_cast<int>(constraints->m_pin_inds.size()); i++) {
        // Construct Selection Matrix to select x
        int pin_id = constraints->m_pin_inds[i];
        S_fix.coeffRef(pin_id, count) = 1.0;
        m_positive_pin(pin_id) = 0;

        count++;
    }

    count = 0;
    for (int i = 0; i < m_positive_pin.size(); ++i) {
        if (m_positive_pin(i) > 0) {  // Free point
            S_free.coeffRef(i, count) = 1.0;
            count++;
        }
    }

    m_S_fix = S_fix;
    m_S_free = S_free;
}


void AlgorithmData::print_initial_data() const {  // X error
    std::cout << "\nADMM it |  time  |     objective    |obj.resid.|";
    if (m_soln_provided) {
        std::cout << " X error  |";
    }
    std::cout << "flips| RW? |";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << "LS:DeltaAL|GS:DeltaAL| L-BFGS ";
    }
    std::cout << std::endl;
    std::cout << "   init" << " |"
              << std::fixed
              << std::setprecision(3)
              << std::setw(7)
              << m_accumulated_time_s[0] << " | "
              << std::scientific
              << std::setprecision(10)
              << m_objectives[0] << " | "
              << "--------" << " | ";
    if (m_soln_provided) {
        std::cout << std::scientific
                  << std::setprecision(2)
                  << m_state_x_errors[0] << " | ";
    }
    std::cout << std::setw(3)
              << m_flips_count[0] << " | "
              << std::setw(3)
              << m_reweighted[0] << " | ";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << "--------" << " | "
              << "--------" << " | "
              << "-----";
    }
    std::cout << std::endl << std::fixed;
}

void AlgorithmData::print_curr_iter_data() const {
    std::cout << std::setw(7) 
              << m_iter << " |"
              << std::fixed
              << std::setprecision(3)
              << std::setw(7)
              << m_accumulated_time_s[m_iter+1] << " | "
              << std::scientific
              << std::setprecision(10)
              << m_objectives[m_iter+1] << " | "
              << std::scientific
              << std::setprecision(2)
              << m_norm_obj_residuals[m_iter] << " | ";
    if (m_soln_provided) {
        std::cout << std::scientific
                  << std::setprecision(2)
                  << m_state_x_errors[m_iter+1] << " | ";
    }
    std::cout << std::setw(3)
              << m_flips_count[m_iter+1] << " | "
              << std::setw(3)
              << m_reweighted[m_iter+1] << " | ";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << std::setprecision(2)
                  << m_per_iter_ls_delta_auglag[m_iter] << " | "
                  << m_per_iter_gs_delta_auglag[m_iter] << " |"
                  << std::setw(4)
                  << m_per_iter_inner_iters[m_iter];
    }
    std::cout << std::endl << std::fixed;
}

void AlgorithmData::print_final_iter_data() const {
    const int num_iters = m_objectives.size() - 1;
    int i = num_iters-1;
    std::cout << std::setw(7) 
              << i << " |"
              << std::fixed
              << std::setprecision(3)
              << std::setw(7)
              << m_accumulated_time_s[i+1] << " | "
              << std::scientific
              << std::setprecision(10)
              << m_objectives[i+1] << " | "
              << std::scientific
              << std::setprecision(2)
              << m_norm_obj_residuals[i] << " | ";
    if (m_soln_provided) {
        std::cout << std::scientific
                  << std::setprecision(2)
                  << m_state_x_errors[i+1] << " | ";
    }
    std::cout << std::setw(3)
              << m_flips_count[i+1] << " | "
              << std::setw(3)
              << m_reweighted[i+1] << " | ";
    if (m_rot_awareness == Settings::RotAwareness::ENABLED) {
        std::cout << std::setprecision(2)
                  << m_per_iter_ls_delta_auglag[i] << " | "
                  << m_per_iter_gs_delta_auglag[i] << " |"
                  << std::setw(4)
                  << m_per_iter_inner_iters[i];
    }
    std::cout << std::endl << std::fixed;
}

}  // namespace wrapd
