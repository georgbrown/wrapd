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

#ifndef SRC_ELEMENTS_HPP_
#define SRC_ELEMENTS_HPP_

#include <iostream>
#include <memory>
#include <vector>

#include "Lame.hpp"
#include "Math.hpp"
#include "Settings.hpp"

namespace wrapd {

class Elements {
 public:
    Elements(const int num_elements)
            : m_num_elements(num_elements) {
        m_weight_squared.resize(num_elements);
        m_cand_weight_squared.resize(num_elements);
    }

    virtual ~Elements() {}

    int num_elements() const { return m_num_elements; }

    virtual int is_inverted(
            const math::MatX3 &x_free,
            const math::MatX3 &x_fix) {
        (void)(x_free);
        (void)(x_fix);
        return 0;
    }

    double max_wsq_ratio() {
        double max_ratio = 0.;
        for (int e = 0; e < m_num_elements; e++) {
            double currsq_over_candsq = m_weight_squared[e] / m_cand_weight_squared[e];
            double candsq_over_currsq = 1.0 / currsq_over_candsq;
            if (currsq_over_candsq > max_ratio) {
                max_ratio = currsq_over_candsq;
            }
            if (candsq_over_currsq > max_ratio) {
                max_ratio = candsq_over_currsq;
            }
        }
        return max_ratio;
    }

    virtual void update_actual_weights() {
        throw std::runtime_error("Error: Need to implement update_actual_weights");
    }

    virtual void initialize_dual_vars(const math::MatX3& X) {
        (void)(X);
        throw std::runtime_error("Error: Need to implement initialize_dual_vars");
    }

    virtual void advance_dual_vars() {
        throw std::runtime_error("Error: Need to implement advance_dual_vars");
    }

    virtual void midupdate_weights(
            const math::MatX3 &X) {
        (void)(X);
        throw std::runtime_error("Error: Need to implement midupdate_weights(X)");
    }

    virtual void midupdate_weights() {
        throw std::runtime_error("Error: Need to implement midupdate_weights()");
    }

    virtual void update(bool update_candidate_weights) {
        (void)(update_candidate_weights);
        throw std::runtime_error("Error: Need to implement EnergyTerm::update(MatX3, bool)");
    }

    virtual double polar_update(
            bool update_candidate_weights) {
        (void)(update_candidate_weights);
        throw std::runtime_error("Error: Implement polar_update(bool)");
    }

    virtual math::MatX3 add_to_b() const {
        throw std::runtime_error("Error: Need to implement add_to_b()");
    }

    virtual void get_A_triplets(math::Triplets &triplets) {
        (void)(triplets);
        throw std::runtime_error("Error: Need to implement get_A_triplets(triplets)");
    }

    virtual void update_A_coeffs(math::SpMat &A) {
        (void)(A);
        throw std::runtime_error("Error: Need to implement update_A_coeffs(A)");
    }


    virtual double penalty_energy(bool lagged_u) const {
        (void)(lagged_u);
        throw std::runtime_error("Error: Need to implement penalty_energy(lagged_u)");
    }

    virtual double gradient(const math::MatX3 &Ft, math::MatX3 &grad) {
        (void)(Ft);
        (void)(grad);
        throw std::runtime_error("Error: Need to implement gradient");
    }

    virtual double primal_residual_sq(Settings::RotAwareness rot_awareness) const {
        (void)(rot_awareness);
        throw std::runtime_error("Error: Need to implement primal_residual_sq");
    }

    virtual double potential_energy_x() const {
        throw std::runtime_error("Error: Need to implement potential_energy_x");
    }

    virtual double polar_potential_energy() const {
        throw std::runtime_error("Error: Need to implement polar_potential_energy");
    }

 protected:

    virtual double potential_energy(const math::MatX3 &Ft) {
        (void)(Ft);
        throw std::runtime_error("Error: Need to implement EnergyTerm::potential_energy(const MatX3 &Ft)");
    }

    const int m_num_elements;
    std::vector<double> m_weight_squared;
    std::vector<double> m_cand_weight_squared;

    double m_curr_pe;  // last local step delta auglag

};  // end class Elements

}  // namespace wrapd

#endif  // SRC_ELEMENTS_HPP_