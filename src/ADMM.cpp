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

#include <algorithm>
#include <vector>

#include "ADMM.hpp"

namespace wrapd {

ADMM::ADMM() {
    m_initialized = false;
}

bool ADMM::initialize(std::shared_ptr<System> sys, const Settings& settings) {
    m_settings = settings;
    m_system = sys;

    m_algorithm_data = std::make_shared<AlgorithmData>(m_settings, m_system);
    m_algorithm_data->m_runtime = RuntimeData();  // reset

    m_micro_timer.reset();
    m_algorithm_data->m_init_x_fix = m_algorithm_data->m_S_fix.transpose() * sys->X_init();
    m_algorithm_data->m_curr_x_fix = m_algorithm_data->m_init_x_fix;
    m_system->update_fix_cache(m_algorithm_data->m_curr_x_fix);
    if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        m_global_step_behavior = std::make_unique<UnawareGS>(m_settings, m_system, m_algorithm_data);
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        m_global_step_behavior = std::make_unique<RotAwareGS>(m_settings, m_system, m_algorithm_data); 
    } else {
        throw std::runtime_error("Error: Invalid rot_awareness in ADMM::initialize.");
    }
    m_algorithm_data->m_runtime.setup_solve_ms += m_micro_timer.elapsed_ms();

    m_initialized = true;
    return true;
}


void ADMM::solve(math::MatX3& X) {
    m_micro_timer.reset();
    setup_solve(X);
    m_algorithm_data->m_runtime.setup_solve_ms += m_micro_timer.elapsed_ms();

    if ((m_settings.m_reweighting == Settings::Reweighting::ENABLED)) {
        m_global_step_behavior->initialize_weights();
    }

    if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        m_micro_timer.reset();
        m_algorithm_data->m_init_auglag_obj = m_system->auglag_objective(false);
        m_algorithm_data->m_runtime.global_ms += m_micro_timer.elapsed_ms();
    }

    for (int s_i = 0; s_i < m_settings.m_max_admm_iters; ++s_i) {
        iterate();
        if (converged()) {
            break;
        }
    }

    m_micro_timer.reset();
    finish_solve(X);
    m_algorithm_data->m_runtime.finish_solve_ms += m_micro_timer.elapsed_ms();
}




void ADMM::setup_solve(math::MatX3& X) {

    m_algorithm_data->m_curr_x_free = m_algorithm_data->m_S_free.transpose() * X;
    m_algorithm_data->m_soln_x_free = m_algorithm_data->m_S_free.transpose() * m_system->X_soln();

    m_system->update_fix_cache(m_algorithm_data->m_curr_x_fix);
    m_system->update_defo_cache(m_algorithm_data->m_curr_x_free);
    m_system->initialize_dual_vars(X);
    m_algorithm_data->m_prev_x_free = m_algorithm_data->m_curr_x_free;

    m_algorithm_data->m_iter = 0;
    m_algorithm_data->log_initial_data();


    if (m_settings.m_io.verbose()) {
        m_algorithm_data->print_initial_data();
    }

}

void ADMM::iterate() {
    m_micro_timer.reset();
    m_algorithm_data->m_prev_x_free = m_algorithm_data->m_curr_x_free;
    m_system->advance_dual_vars();
    m_algorithm_data->m_runtime.setup_solve_ms += m_micro_timer.elapsed_ms();


    // Local step
    bool update_candidate_weights = (m_settings.m_reweighting == Settings::Reweighting::ENABLED);
    if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        m_micro_timer.reset();
        m_system->local_update(update_candidate_weights);
        m_algorithm_data->m_runtime.local_ms += m_micro_timer.elapsed_ms();
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        m_micro_timer.reset();
        m_algorithm_data->m_last_localstep_delta_auglag_obj = m_system->polar_local_update(update_candidate_weights);
        m_algorithm_data->m_runtime.local_ms += m_micro_timer.elapsed_ms();
    }


    m_system->update_defo_cache(m_algorithm_data->m_curr_x_free);  // TEMP(George)

    // Global step (records runtime internally)
    m_global_step_behavior->global_step();
    double m_last_globalstep_delta_auglag_obj = m_global_step_behavior->last_step_delta_auglag();


    // Reweighting step
    m_micro_timer.reset();
    bool reweighted = false;
    if (m_settings.m_reweighting == Settings::Reweighting::ENABLED) {
        reweighted = m_global_step_behavior->update_weights();
    }
    m_algorithm_data->m_runtime.refactor_ms += m_micro_timer.elapsed_ms();

    ///////////////

    m_algorithm_data->update_per_iter_reweighted(reweighted);
    m_algorithm_data->update_per_iter_delta_auglag(m_algorithm_data->m_last_localstep_delta_auglag_obj, m_last_globalstep_delta_auglag_obj);
    m_algorithm_data->update_per_iter_data();
    m_algorithm_data->update_per_iter_runtime();
    if (m_settings.m_io.verbose() > 0 && m_algorithm_data->m_iter % 50 == 0) {
        m_algorithm_data->print_curr_iter_data();
    }

    m_algorithm_data->m_iter += 1;

}

bool ADMM::converged() {
    bool no_recent_flips = m_algorithm_data->no_recent_flips();
    double curr_norm_obj_residual = m_algorithm_data->current_normalized_objective_residual();

    if (no_recent_flips && curr_norm_obj_residual < m_settings.m_tol) {
        return true;
    } else if (m_settings.m_io.soln_provided() && m_algorithm_data->current_normalized_x_error() < 1.e-4) {
        return true;
    } else {
        return false;
    }
}

void ADMM::finish_solve(math::MatX3& X) {

    // Setting the final x solution
    X = (m_algorithm_data->m_S_free * m_algorithm_data->m_curr_x_free) + (m_algorithm_data->m_S_fix * m_algorithm_data->m_curr_x_fix);


    if (m_settings.m_io.verbose() > 0 && (m_algorithm_data->m_iter-1) % 50 != 0) {
        m_algorithm_data->print_final_iter_data();
        std::cout << std::endl;
    }

}

}  // namespace wrapd
