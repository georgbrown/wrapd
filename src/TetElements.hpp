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

#ifndef SRC_TETELEMENTS_HPP_
#define SRC_TETELEMENTS_HPP_

#include <memory>
#include <vector>

#include "Elements.hpp"

#include "MCL/Newton.hpp"
#include "MCL/LBFGS.hpp"


namespace wrapd {

class Prox : public mcl::optlib::Problem<double, 3> {
 public:
    virtual void set_x0(const math::Vec3 x0) = 0;  // set x0 (quad penalty)
    virtual bool converged(const math::Vec3 &x0, const math::Vec3 &x1, const math::Vec3 &grad) {
        return ( grad.norm() < 1e-8 || (x0-x1).norm() < 1e-8 );
    }
    virtual double physical_value(const math::Vec3 &F) = 0;
};


// Neo-Hookean model
// This extended-log formulation is the same one used by Liu et al. 2017
// Other NH models should work too. We matched theirs so we could make comparisons in the paper.
class NHProx : public Prox {
 public:
    void set_lame(const Lame &lame) {
        m_mu = lame.mu();
        m_lambda = lame.lambda();
        m_k = 2.0066 * m_mu + 1.0122 * m_lambda;
        m_J0 = 0.1;
    }
    void set_x0(const math::Vec3 x0) { m_x0 = x0; }
    void set_wsq_over_volume(double wsq_over_volume) { m_wsq_over_volume = wsq_over_volume; }
    
    double energy_density(const math::Vec3 &x) const {
        if (x[0] < 0. || x[1] < 0. || x[2] < 0.) {
            return 1.e15;
        }
        double I_1 = x.squaredNorm();
	    double J = x.prod();

        double e = 0.5 * m_mu * (I_1 - 3.0);
        if (J > m_J0) {
            double logJ = std::log(J);
            e += -m_mu*logJ + 0.5*m_lambda*logJ*logJ;
        } else {
            double fJ = std::log(m_J0) + ((J - m_J0) / m_J0) - (0.5 * std::pow(((J / m_J0) - 1.0), 2.0));
            e += -m_mu*fJ + 0.5*m_lambda*fJ*fJ;
        }
        return e;
    }

    double physical_value(const math::Vec3 &x) {
        if (x[0] < 0. || x[1] < 0. || x[2] < 0.) {
            return 1.e15;
        }
        double e = energy_density(x) - energy_density(math::Vec3::Ones());
        return e;
    }

    double value(const math::Vec3 &x) {
        double t1 = energy_density(x);
        double t2 = (m_wsq_over_volume * 0.5) * (x - m_x0).squaredNorm();  // quad penalty
        return t1 + t2;
    }

    double gradient(const math::Vec3 &x, math::Vec3 &grad) {
        double J = x.prod();
	    math::Vec3 x_inv = x.cwiseInverse();

        if (J > m_J0) {
	        grad = m_mu*x + (m_lambda*std::log(J) - m_mu) * x_inv;
        } else {
            double fJ = std::log(m_J0) + ((J - m_J0) / m_J0) - (0.5 * std::pow(((J / m_J0) - 1.0), 2.0));
            double dfJdJ = 1.0 / m_J0 - (J - m_J0) / (m_J0*m_J0);
            grad = m_mu*x + (m_lambda*fJ - m_mu) * dfJdJ * J * x_inv;
        }

        grad += m_wsq_over_volume * (x - m_x0);
        return value(x);
    }

    void hessian_density(const math::Vec3 &x, math::Mat3x3 &hess) {
        double J = x.prod();

        if (J > m_J0) {
            double logJ = std::log(J);
            double common_term = m_mu - m_lambda * (logJ - 1.0);
            hess(0, 0) = m_mu + common_term / (x[0]*x[0]);
            hess(1, 1) = m_mu + common_term / (x[1]*x[1]);
            hess(2, 2) = m_mu + common_term / (x[2]*x[2]);
            hess(0, 1) = hess(1, 0) = m_lambda / (x[0] * x[1]);
            hess(0, 2) = hess(2, 0) = m_lambda / (x[0] * x[2]);
            hess(1, 2) = hess(2, 1) = m_lambda / (x[1] * x[2]);
        } else {
            double fJ = std::log(m_J0) + ((J - m_J0) / m_J0) - (0.5 * std::pow(((J / m_J0) - 1.0), 2.0));
            double dfJdJ = 1.0 / m_J0 - (J - m_J0) / (m_J0*m_J0);

            double lambda_fJ_minus_mu = m_lambda * fJ - m_mu;

            hess(0, 0) = m_mu - (lambda_fJ_minus_mu) * std::pow(x[1] * x[2] / m_J0, 2.0)
                    + m_lambda * std::pow(dfJdJ * x[1] * x[2] , 2.0);

            hess(1, 1) = m_mu - (lambda_fJ_minus_mu) * std::pow(x[0] * x[2] / m_J0, 2.0)
                    + m_lambda * std::pow(dfJdJ * x[0] * x[2] , 2.0);

            hess(2, 2) = m_mu - (lambda_fJ_minus_mu) * std::pow(x[0] * x[1] / m_J0, 2.0)
                    + m_lambda * std::pow(dfJdJ * x[0] * x[1] , 2.0);

            double common_term = lambda_fJ_minus_mu * (dfJdJ - J/(m_J0*m_J0)) + m_lambda * J * std::pow(dfJdJ, 2.0); 

            hess(0, 1) = hess(1, 0) = x[2] * common_term;
            hess(0, 2) = hess(2, 0) = x[1] * common_term;
            hess(1, 2) = hess(2, 1) = x[0] * common_term;

        }
        math::makePD(hess);
    }

    void hessian(const math::Vec3 &x, math::Mat3x3 &hess) {
        hessian_density(x, hess);
        hess += m_wsq_over_volume * math::Mat3x3::Identity();
    }

    double physical_stiffness() const { return m_k; }
    double mu() const { return m_mu; }
    double lambda() const { return m_lambda; }

 private:
    double m_mu;
    double m_lambda;
    double m_J0;
    double m_k;
    double m_wsq_over_volume;
    math::Vec3 m_x0;
};


class TetElements : public Elements {
 public:
    TetElements(
            const std::vector<math::Vec4i> &tet,
            const std::vector<math::Vec4i> &free_index_tet,
            const std::vector<math::Vec4i> &fix_index_tet,
            const std::vector< std::vector<math::Vec3> > &verts,
            const Settings &settings);

    virtual double penalty_energy(bool lagged_u) const final;

    virtual double primal_residual_sq(Settings::RotAwareness rot_awareness) const final;

    void update_fix_cache(const math::MatX3 &x_fix);
    virtual void update_defo_cache(const math::MatX3 &x_free, bool compute_polar_data = true) final;

    virtual double potential_energy_x() const;

    virtual double polar_potential_energy() const final;

    virtual void initialize_dual_vars(const math::MatX3& X) final;
    virtual void advance_dual_vars() final;

    virtual void update(bool update_candidate_weights) final;
    virtual double polar_update(bool update_candidate_weights) final;

    virtual math::MatX3 add_to_b() const final;
    virtual void get_A_triplets(math::Triplets &triplets) final;
    virtual void update_A_coeffs(math::SpMat &A) final;

    virtual double global_obj_value(const math::MatX3 &x_free) final;
    virtual double global_obj_grad(const math::MatX3 &x_free, math::MatX3 &gradient) final;
    math::MatX3 global_obj_grad() const;

    virtual void midupdate_weights(const math::MatX3 &X);
    virtual void midupdate_weights();

    int inversion_count() const;

    void update_all_candidate_weights(const std::vector<math::Vec3>& temp_sigmas);

    virtual void update_actual_weights() final;

    const std::vector<math::Vec4i>& inds() const { return m_tet; }

 protected:
    std::vector<math::Vec4i> m_tet;
    std::vector<math::Vec4i> m_free_index_tet;
    std::vector<math::Vec4i> m_fix_index_tet;

    const Settings& m_settings;
    const ModelSettings& m_model_settings;

    std::vector<double> m_volume;
    std::vector<math::Mat3x3> m_edges_inv;

    std::vector<math::Mat3x4> m_Di_local;
    std::vector<math::Mat3X> m_Dfree_local;
    std::vector<math::Mat3X> m_Dfix_local;

    std::vector<math::Mat3x3> m_Dfix_xfix_local;
    std::vector<math::Mat4x4> m_Dt_D_local;
    std::vector<math::Vec3> m_last_sigma;

    std::vector<math::Mat3x3> m_Zi; 
    std::vector<math::Mat3x3> m_prev_Zi;  
    std::vector<math::Mat3x3> m_Ui;
    std::vector<math::Mat3x3> m_prev_Ui;

    // Cached data - be careful to ensure the cache isn't stale!
    std::vector<math::Mat3x3> m_cached_F;
    std::vector<math::Mat3x3> m_cached_symF;
    std::vector<math::Vec3> m_cached_svdsigma;
    std::vector<math::Mat3x3> m_cached_svdU;
    std::vector<math::Mat3x3> m_cached_svdV;
    int m_xfree_rows;

    std::vector<int> m_num_free_inds;

    mcl::optlib::LSMethod m_elastic_prox_ls_method;


};  // end class TetElements


template <typename IN_SCALAR, typename TYPE>
inline void create_tets_from_mesh(
        std::shared_ptr<TetElements>& tet_elements,
        const IN_SCALAR *verts,
        const int *inds,
        int n_tets,
        const Settings& settings,
        const std::vector<int> &free_index_list,
        const std::vector<int> &fix_index_list,
        const int vertex_offset) {

    std::vector<math::Vec4i> tets(n_tets);
    std::vector<math::Vec4i> free_index_tets(n_tets);
    std::vector<math::Vec4i> fix_index_tets(n_tets);
    std::vector< std::vector<math::Vec3> > tets_verts(n_tets);

    // TODO: Parallelize this
    for (int e = 0; e < n_tets; ++e) {
        tets[e] = math::Vec4i(inds[e*4+0], inds[e*4+1], inds[e*4+2], inds[e*4+3]);
        free_index_tets[e] = math::Vec4i(
                free_index_list[inds[e*4+0]],
                free_index_list[inds[e*4+1]],
                free_index_list[inds[e*4+2]],
                free_index_list[inds[e*4+3]]);
        fix_index_tets[e] = math::Vec4i(
                fix_index_list[inds[e*4+0]],
                fix_index_list[inds[e*4+1]],
                fix_index_list[inds[e*4+2]],
                fix_index_list[inds[e*4+3]]);
        std::vector<math::Vec3> tet_verts = {
            math::Vec3(verts[tets[e][0]*3+0], verts[tets[e][0]*3+1], verts[tets[e][0]*3+2]),
            math::Vec3(verts[tets[e][1]*3+0], verts[tets[e][1]*3+1], verts[tets[e][1]*3+2]),
            math::Vec3(verts[tets[e][2]*3+0], verts[tets[e][2]*3+1], verts[tets[e][2]*3+2]),
            math::Vec3(verts[tets[e][3]*3+0], verts[tets[e][3]*3+1], verts[tets[e][3]*3+2])};
        tets_verts[e] = tet_verts;
        tets[e] += (math::Vec4i(1, 1, 1, 1) * vertex_offset);
    }
    tet_elements = std::make_shared<TYPE>(tets, free_index_tets, fix_index_tets, tets_verts, settings);
}  // end create from mesh


}  // namespace wrapd
#endif  // SRC_TETELEMENTS_HPP_
