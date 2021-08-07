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

#include "System.hpp"

namespace wrapd {

System::System(const Settings &settings)
    : m_settings(settings) {
    m_constraints = std::make_shared<ConstraintSet>(ConstraintSet());
    m_initialized = false;
}

void System::set_pins(
        const std::vector<int> &inds,
        const std::vector<math::Vec3> &points) {
    int n_pins = inds.size();
    const int dof = 3 * m_X.rows();
    bool pin_in_place = static_cast<int>(points.size()) != n_pins;
    if ( (dof == 0 && pin_in_place) || (pin_in_place && points.size() > 0) ) {
        throw std::runtime_error("**Solver::set_pins Error: Bad input.");
    }
    m_constraints->m_pin_inds = inds;
    m_xfree_rows = m_X.rows() - inds.size();
}

std::shared_ptr<ConstraintSet> System::constraint_set() {
    return m_constraints;
}

bool System::possibly_update_weights(double gamma) {
    const int n_tri_elements = num_tet_elements();
    double do_reweight = false;

    if (n_tri_elements > 0) {
        double max_wsq_ratio = m_tet_elements->max_wsq_ratio();
        do_reweight = (max_wsq_ratio > gamma);
        if (do_reweight) {
            m_tet_elements->update_actual_weights();
        }
    }

    return do_reweight;
}


math::MatX3 System::get_b() {
    const int n_tet_elements = num_tet_elements();

    math::MatX3 b = math::MatX3::Zero(m_xfree_rows, 3);

    if (n_tet_elements > 0) {
        b += m_tet_elements->add_to_b();
    }

    return b;
}

void System::get_A(math::SpMat &A) {
    const int n_tet_elements = num_tet_elements();

    math::Triplets triplets_all;

    if (n_tet_elements > 0) {
        math::Triplets triplets_tets;
        m_tet_elements->get_A_triplets(triplets_tets);
        triplets_all.insert(std::end(triplets_all), std::begin(triplets_tets), std::end(triplets_tets));
    }

    A.resize(m_xfree_rows, m_xfree_rows);
    A.setFromTriplets(triplets_all.begin(), triplets_all.end());
}

void System::update_A(math::SpMat &A) {
    // Note: Do not setZero here! Space has already been reserved, no zeroing is needed.
    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->update_A_coeffs(A);
    }
}


double System::penalty_energy(bool lagged_u) const {
    double se = 0.;

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        se += m_tet_elements->penalty_energy(lagged_u);        
    }

    return se;
}


void System::midupdate_WtW(
        const math::MatX3& X) {

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->midupdate_weights(X);        
    }
}

void System::midupdate_WtW() {

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->midupdate_weights();    
    }
}


int System::inversion_count() const {
    int inv_count = 0;
    const int n_tet_elements = num_tet_elements();

    // We only check for flipped tetrahedra in 3D simulations.
    if (n_tet_elements > 0) {
        inv_count += m_tet_elements->inversion_count();
    }

    return inv_count;    
}

void System::local_update(bool update_candidate_weights) {

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->update(update_candidate_weights);        
    }
}

double System::polar_local_update(
        bool update_candidate_weights) {
    double delta_auglag = 0;

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        delta_auglag += m_tet_elements->polar_update(update_candidate_weights);
    }

    return delta_auglag;
}


void System::initialize() {
    m_initialized = true;
}


void System::update_fix_cache(const math::MatX3 &x_fix) {
    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->update_fix_cache(x_fix);
    } 
}

void System::update_defo_cache(const math::MatX3 &x_free, bool update_polar_data) {
    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        m_tet_elements->update_defo_cache(x_free, update_polar_data);
    }
}

void System::initialize_dual_vars(const math::MatX3 &X) {
    const int n_tet_elements = num_tet_elements();
    if (n_tet_elements > 0) {
        m_tet_elements->initialize_dual_vars(X);   
    }          
}

void System::advance_dual_vars() {
    const int n_tet_elements = num_tet_elements();
    if (n_tet_elements > 0) {
        m_tet_elements->advance_dual_vars();
    }      
}


double System::primal_residual() const {
    const int n_tet_elements = num_tet_elements();

    double res_sq = 0.;

    if (n_tet_elements > 0) {
        res_sq += m_tet_elements->primal_residual_sq(m_settings.m_rot_awareness);
    }

    return std::sqrt(res_sq);
}

double System::potential_energy() const {
    double pe = 0.;

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        pe += m_tet_elements->potential_energy_x();        
    }

    return pe;
}

double System::polar_potential_energy() const {
    double pe = 0.;

    const int n_tet_elements = num_tet_elements();


    if (n_tet_elements > 0) {
        pe += m_tet_elements->polar_potential_energy();
    }

    return pe;
}


double System::global_obj_value(const math::MatX3 &x_free) {
    double value = 0.;

    const int n_tet_elements = num_tet_elements();

    if (n_tet_elements > 0) {
        value += m_tet_elements->global_obj_value(x_free);
    }

    return value;
}

double System::global_obj_grad(
        const math::MatX3 &x_free,
        math::MatX3 &grad,
        bool positions_have_changed) {

    const int n_tet_elements = num_tet_elements();

    grad = math::MatX3::Zero(m_xfree_rows, 3);
    double value = 0.;

    if (positions_have_changed) {

        if (n_tet_elements > 0) {
            math::MatX3 grad_tets;
            value += m_tet_elements->global_obj_grad(x_free, grad_tets);
            grad += grad_tets;        
        }

    } else {

        if (n_tet_elements > 0) {
            grad += m_tet_elements->global_obj_grad();       
        }   
    }
    return value;
}


double System::problem_objective() {
    double value = 0.;
    value += potential_energy();
    return value;
}

double System::auglag_objective(bool lagged_u) {
    double pe = polar_potential_energy();
    double se = penalty_energy(lagged_u);
    double total = pe + se;
    return total;    
}

}  // namespace wrapd
