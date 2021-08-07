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
#include <iomanip>

#include "TetElements.hpp"
#include "MCL/MicroTimer.hpp"

namespace wrapd {

//
// TetElements
//

TetElements::TetElements(
        const std::vector<math::Vec4i> &tet,
        const std::vector<math::Vec4i> &free_index_tet,
        const std::vector<math::Vec4i> &fix_index_tet,
        const std::vector<std::vector<math::Vec3> > &verts,
        const Settings &settings) :
        Elements(tet.size()),
        m_tet(tet),
        m_free_index_tet(free_index_tet),
        m_fix_index_tet(fix_index_tet),
        m_settings(settings),
        m_model_settings(settings.m_model_settings) {

    if (m_model_settings.elastic_prox_ls_method() == ModelSettings::ProxLSMethod::Backtracking) {
        m_elastic_prox_ls_method = mcl::optlib::LSMethod::Backtracking;
    } else if (m_model_settings.elastic_prox_ls_method() == ModelSettings::ProxLSMethod::BacktrackingCubic) {
        m_elastic_prox_ls_method = mcl::optlib::LSMethod::BacktrackingCubic;
    } else {
        throw std::runtime_error("Error: Invalid prox ls method in TetElements constructor.");
    }

    m_volume.resize(m_num_elements);
    m_edges_inv.resize(m_num_elements);

    m_Di_local.resize(m_num_elements);
    m_Dfree_local.resize(m_num_elements);
    m_Dfix_local.resize(m_num_elements);
    
    m_Dfix_xfix_local.resize(m_num_elements);
    m_Dt_D_local.resize(m_num_elements);
    m_last_sigma.resize(m_num_elements);

    m_Zi.resize(m_num_elements);
    m_prev_Zi.resize(m_num_elements);
    m_Ui.resize(m_num_elements);
    m_prev_Ui.resize(m_num_elements);

    m_cached_F.resize(m_num_elements);
    m_cached_symF.resize(m_num_elements);
    m_cached_svdsigma.resize(m_num_elements);
    m_cached_svdU.resize(m_num_elements);
    m_cached_svdV.resize(m_num_elements);
    m_num_free_inds.resize(m_num_elements);

    double k;
    if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::ARAP) {
        k = 1.;
    } else if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::NH) {
        k = 2.0066 * m_model_settings.elastic_lame().mu() + 1.0122 * m_model_settings.elastic_lame().lambda();
    } else {
        throw std::runtime_error("Error: Unsupported elastic model in TetElements constructor.");
    }
    double wsq_mult = m_model_settings.elastic_beta_static();

    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {

        math::Mat3x3 edges;
        edges.col(0) = verts[e][1] - verts[e][0];
        edges.col(1) = verts[e][2] - verts[e][0];
        edges.col(2) = verts[e][3] - verts[e][0];
        m_edges_inv[e] = edges.inverse();
        m_volume[e] = (edges).determinant() / 6.0;
        if (m_volume[e] < 0.) {
            throw std::runtime_error("**Error: Inverted tet in TetElements constructor.");
        }

        m_num_free_inds[e] = 0;
        for (int i = 0; i < 4; i++) {
            if (m_free_index_tet[e][i] >= 0) {
                m_num_free_inds[e] += 1;
            }
        }

        math::Mat4x3 S;
        S.setZero();
        S(0, 0) = -1;
        S(0, 1) = -1;
        S(0, 2) = -1;
        S(1, 0) =  1;
        S(2, 1) =  1;
        S(3, 2) =  1;
        m_Di_local[e] = (S * m_edges_inv[e]).transpose();
        m_Dt_D_local[e] = m_Di_local[e].transpose() * m_Di_local[e];
        m_Dfree_local[e].resize(3, m_num_free_inds[e]);
        m_Dfix_local[e].resize(3, 4 - m_num_free_inds[e]);

        int j = 0;
        for (int i = 0; i < 4; i++) {
            if (m_free_index_tet[e][i] >= 0) {
                m_Dfree_local[e].col(j) = m_Di_local[e].col(i);
                j += 1;
            }
        }

        j = 0;
        for (int i = 0; i < 4; i++) {
            if (m_fix_index_tet[e][i] >= 0) {
                m_Dfix_local[e].col(j) = m_Di_local[e].col(i);
                j += 1;
            }
        }
 
        m_weight_squared[e] = wsq_mult * k * m_volume[e];
        m_last_sigma[e] = math::Vec3::Ones();

    }
}

void TetElements::initialize_dual_vars(const math::MatX3& X) {

    if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat4x3 x_local;
            x_local.row(0) = X.row(m_tet[e][0]);
            x_local.row(1) = X.row(m_tet[e][1]);
            x_local.row(2) = X.row(m_tet[e][2]);
            x_local.row(3) = X.row(m_tet[e][3]);
            math::Mat3x3 Ft = m_Di_local[e] * x_local;
            math::Mat3x3 F = Ft.transpose();
            math::Mat3x3 R;
            math::Mat3x3 S;
            math::polar(F, R, S, true);

            m_Zi[e] = S.transpose();
            m_prev_Zi[e] = m_Zi[e];
            m_Ui[e].setZero();
            m_prev_Ui[e].setZero();
        }
        m_curr_pe = polar_potential_energy();
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat4x3 x_local;
            x_local.row(0) = X.row(m_tet[e][0]);
            x_local.row(1) = X.row(m_tet[e][1]);
            x_local.row(2) = X.row(m_tet[e][2]);
            x_local.row(3) = X.row(m_tet[e][3]);
            m_Zi[e] = m_Di_local[e] * x_local;
            m_prev_Zi[e] = m_Zi[e];
            m_Ui[e].setZero();
            m_prev_Ui[e].setZero();
        }        
    } else {
        throw std::runtime_error("Error: Invalid rot_awareness value in TetElements::initialize_dual_vars.");
    }
}

void TetElements::advance_dual_vars() {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        m_prev_Zi[e] = m_Zi[e];
        m_prev_Ui[e] = m_Ui[e];    
    }
}

void TetElements::update(bool update_candidate_weights) {
    std::vector<math::Vec3> temp_sigmas(m_num_elements);

    if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::ARAP) {

        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x3 Di_x = m_cached_F[e].transpose();
            math::Mat3x3 Ft = Di_x + m_Ui[e];

            math::Mat3x3 F = Ft.transpose();
            math::Mat3x3 R = math::rot(F);

            Ft.noalias() = 0.5 * (F + R).transpose();

            m_Ui[e] += (Di_x - Ft);
            m_Zi[e] = Ft;
        }
    } else if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::NH) {

        #pragma omp parallel
        {
            // mcl::optlib::Newton<double, 3> solver;
            mcl::optlib::LBFGS<double, 3> solver; // Seems to be faster than Newton, but either should work
            solver.m_settings.ls_method = m_elastic_prox_ls_method;
            solver.m_settings.max_iters = m_model_settings.elastic_prox_iters();
            NHProx problem;
            problem.set_lame(m_model_settings.elastic_lame());
            
            #pragma omp for
            for (int e = 0; e < m_num_elements; e++) {
                math::Mat3x3 Di_x = m_cached_F[e].transpose();
                math::Mat3x3 Ft = Di_x + m_Ui[e];

                math::Mat3x3 F = Ft.transpose();
                math::Vec3 sigma;
                math::Mat3x3 U;
                math::Mat3x3 V;
                math::svd(F, sigma, U, V, true);
                
                problem.set_x0(sigma);
                problem.set_wsq_over_volume(m_weight_squared[e] / m_volume[e]);
                sigma = m_last_sigma[e];
                // sigma = math::Vec3::Ones();
                solver.minimize(problem, sigma);
                m_last_sigma[e] = sigma;
                temp_sigmas[e] = sigma;

                Ft.noalias() = V * sigma.asDiagonal() * U.transpose();

                m_Ui[e] += (Di_x - Ft);
                m_Zi[e] = Ft;
            }
        }
    } else {
        throw std::runtime_error("Error: Invalid model in TetElements::update");
    }

    if (update_candidate_weights) {
        update_all_candidate_weights(temp_sigmas);
    }
}

double TetElements::polar_update(bool update_candidate_weights) {

    double bef_pe = m_curr_pe;
    m_curr_pe = 0.;
    double bef_penalty = 0.;
    double aft_penalty = 0.;

    std::vector<math::Vec3> temp_sigmas(m_num_elements);

    if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::ARAP) {

        double k = 1.; 

        #pragma omp parallel for reduction ( + : m_curr_pe, bef_penalty, aft_penalty)
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x3 S_inout = m_cached_symF[e] + m_Ui[e]; // don't need to use S.transpose() since guaranteed symmetric

            bef_penalty += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (m_Zi[e] - m_Ui[e])).squaredNorm();

            S_inout = 0.5 * (S_inout + math::Mat3x3::Identity());
            m_curr_pe += 0.5 * k * m_volume[e] * (m_cached_symF[e] - math::Mat3x3::Identity()).squaredNorm();

            aft_penalty += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (S_inout - m_Ui[e])).squaredNorm();

            m_Zi[e] = S_inout;
            m_Ui[e] += m_cached_symF[e] - S_inout;
        }
    } else if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::NH) {

        std::vector<math::Mat3x3> S_inout(m_num_elements);
        #pragma omp parallel for reduction ( + : bef_penalty)
        for (int e = 0; e < m_num_elements; e++) {
            S_inout[e] = (m_cached_symF[e] + m_Ui[e]).transpose();
            bef_penalty += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (m_Zi[e] - m_Ui[e])).squaredNorm();
        }

        // NH Specific
        std::vector<math::Vec3> sigma(m_num_elements);
        std::vector<math::Mat3x3> U(m_num_elements);
        std::vector<math::Mat3x3> V(m_num_elements);
        math::svd(S_inout, sigma, U, V);

        double lambda = m_model_settings.elastic_lame().lambda();
        double mu = m_model_settings.elastic_lame().mu();
        double J0 = 0.1;

        #pragma omp parallel
        {
            // mcl::optlib::Newton<double, 3> solver;
            mcl::optlib::LBFGS<double, 3> solver;  // Seems to be faster than Newton, but either should work
            solver.m_settings.ls_method = m_elastic_prox_ls_method;
            solver.m_settings.max_iters = m_model_settings.elastic_prox_iters();
            NHProx problem;
            problem.set_lame(m_model_settings.elastic_lame());

            #pragma omp for reduction ( + : m_curr_pe)
            for (int e = 0; e < m_num_elements; e++) {
                problem.set_x0(sigma[e]);
                problem.set_wsq_over_volume(m_weight_squared[e] / m_volume[e]);
                sigma[e] = m_last_sigma[e];
                // sigma[e] = math::Vec3::Ones();
                solver.minimize(problem, sigma[e]);
                m_last_sigma[e] = sigma[e];
                temp_sigmas[e] = sigma[e];
                if (sigma[e][0] < 0. || sigma[e][1] < 0. || sigma[e][2] < 0.) {
                    printf("Negative singular value in solution. svals: %f %f %f:\n", sigma[e][0], sigma[e][1], sigma[e][2]);                    
                    exit(0);
                }
                S_inout[e].noalias() = V[e] * sigma[e].asDiagonal() * U[e].transpose();

                double I_1 = sigma[e].squaredNorm();
                double J = sigma[e].prod();
                double val = 0.5 * mu * (I_1 - 3.0);
                if (J > J0) {
                    double logJ = std::log(J);
                    val += -mu*logJ + 0.5*lambda*logJ*logJ;
                } else {
                    double fJ = std::log(J0) + ((J - J0) / J0) - (0.5 * std::pow(((J / J0) - 1.0), 2.0));
                    val += -mu*fJ + 0.5*lambda*fJ*fJ;
                }
                m_curr_pe += m_volume[e] * val;

            }
        }

        #pragma omp parallel for reduction( + : aft_penalty)
        for (int e = 0; e < m_num_elements; e++) {
            aft_penalty += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (S_inout[e] - m_Ui[e])).squaredNorm();
            m_Zi[e] = S_inout[e];
            m_Ui[e] += m_cached_symF[e] - S_inout[e];
        }
    } else {
        throw std::runtime_error("Error: Invalid model in TetElements::polar_update");
    }

    if (update_candidate_weights) {
        update_all_candidate_weights(temp_sigmas);
    }

    return (aft_penalty - bef_penalty + m_curr_pe - bef_pe);
}



void TetElements::midupdate_weights(const math::MatX3 &X) {
    if (m_settings.m_reweighting == Settings::Reweighting::ENABLED) {
        std::vector<math::Vec3> temp_sigmas(m_num_elements);
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {

            math::Mat4x3 x_local;
            x_local.row(0) = X.row(m_tet[e][0]);
            x_local.row(1) = X.row(m_tet[e][1]);
            x_local.row(2) = X.row(m_tet[e][2]);
            x_local.row(3) = X.row(m_tet[e][3]);
            math::Mat3x3 Di_x = m_Di_local[e] * x_local;

            math::Mat3x3 F = Di_x.transpose();
            math::Mat3x3 U;
            math::Mat3x3 V;
            math::svd(F, temp_sigmas[e], U, V, true);
        }
        update_all_candidate_weights(temp_sigmas);
        update_actual_weights();
    }
}

void TetElements::midupdate_weights() {
    if (m_settings.m_reweighting == Settings::Reweighting::ENABLED) {
        update_all_candidate_weights(m_cached_svdsigma);
        update_actual_weights();
    }
}


void TetElements::update_all_candidate_weights(const std::vector<math::Vec3>& temp_sigmas) {

    if (m_model_settings.elastic_model() != ModelSettings::ElasticModel::NH) {
        throw std::runtime_error("Error: Adaptive weights are only supported for the NH model.");
    }

    if (m_settings.m_reweighting == Settings::Reweighting::ENABLED) {
        
        // double wsq_rest = 2.0 * m_model_settings.elastic_lame().mu() + m_model_settings.elastic_lame().lambda();
        double wsq_rest = 2.0066 * m_model_settings.elastic_lame().mu() + 1.0122 * m_model_settings.elastic_lame().lambda();
        double wsq_min = wsq_rest * m_model_settings.elastic_beta_min();
        double wsq_max = wsq_rest * m_model_settings.elastic_beta_max();

        static int count = 0;
        if (count > m_settings.m_reweighting_delay) {
            #pragma omp parallel
            {
                NHProx problem;
                problem.set_lame(m_model_settings.elastic_lame());
                math::Mat3x3 K;

                #pragma omp for
                for (int e = 0; e < m_num_elements; e++) {
                    problem.hessian_density(temp_sigmas[e], K);

                    double Kvalmax = K.eigenvalues().real().maxCoeff();
                    double cand_stiffness = math::clamp(wsq_min, Kvalmax, wsq_max);

                    m_cand_weight_squared[e] = m_volume[e] * cand_stiffness;
                }
            }
        } else {
            #pragma omp parallel for
            for (int e = 0; e < m_num_elements; e++) {
                m_cand_weight_squared[e] = m_volume[e] * wsq_rest;
            }
        }
        count++;
    } else {
         throw std::runtime_error("Error: Reweighting is disabled so we shouldn't be updating candidate weights");
    }
}


void TetElements::update_actual_weights() {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        m_prev_Ui[e] = (m_weight_squared[e] / m_cand_weight_squared[e]) * m_prev_Ui[e];
        m_Ui[e] = (m_weight_squared[e] / m_cand_weight_squared[e]) * m_Ui[e];
        m_weight_squared[e] = m_cand_weight_squared[e];
    }
}

int TetElements::inversion_count() const {
    int count = 0;

    if (m_settings.m_rot_awareness == Settings::RotAwareness::ENABLED) {
        #pragma omp parallel for reduction (+:count)
        for (int e = 0; e < m_num_elements; e++) {
            if (m_cached_svdsigma[e][0] < 0. || m_cached_svdsigma[e][1] < 0. || m_cached_svdsigma[e][2] < 0.) {
                count += 1;
            }
        }
    } else if (m_settings.m_rot_awareness == Settings::RotAwareness::DISABLED) {
        #pragma omp parallel for reduction (+:count)
        for (int e = 0; e < m_num_elements; e++) {
            math::Vec3 sigma;
            math::Mat3x3 U;
            math::Mat3x3 V;
            math::svd(m_cached_F[e], sigma, U, V, true);             
            if (sigma[0] < 0. || sigma[1] < 0. || sigma[2] < 0.) {
                count += 1;
            }
        }
    }

    return count;
}

math::MatX3 TetElements::add_to_b() const {
    math::MatX3 b = math::MatX3::Zero(m_xfree_rows, 3);
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat4x3 b_local = m_weight_squared[e] * m_Di_local[e].transpose() * (m_Zi[e] - m_Ui[e] - m_Dfix_xfix_local[e]);
        for (int i = 0; i < 4; i++) {
            if (m_free_index_tet[e][i] >= 0) {
                b.row(m_free_index_tet[e][i]) += b_local.row(i); 
            }
        }
    }
    return b;
}

void TetElements::get_A_triplets(math::Triplets &triplets) {
    triplets.reserve(16*m_num_elements);
    for (int e = 0; e < m_num_elements; e++) {
        for (int i = 0; i < 4; i++) {
            if (m_free_index_tet[e][i] >= 0) {
                for (int j = 0; j < 4; j++) {
                    if (m_free_index_tet[e][j] >= 0 && m_free_index_tet[e][i] <= m_free_index_tet[e][j]) {
                        triplets.emplace_back(math::Triplet(m_free_index_tet[e][i], m_free_index_tet[e][j], m_weight_squared[e] * m_Dt_D_local[e](i, j)));
                    }
                }
            }
        }
    }
}

void TetElements::update_A_coeffs(math::SpMat &A) {
    for (int e = 0; e < m_num_elements; e++) {
        for (int i = 0; i < 4; i++) {
            if (m_free_index_tet[e][i] >= 0) {
                for (int j = 0; j < 4; j++) {
                    if (m_free_index_tet[e][j] >= 0 && m_free_index_tet[e][i] <= m_free_index_tet[e][j]) {
                        A.coeffRef(m_free_index_tet[e][i], m_free_index_tet[e][j]) += m_weight_squared[e] * m_Dt_D_local[e](i, j);
                    }
                }
            }
        }
    }    
}


double TetElements::penalty_energy(bool lagged_u) const {
    double se = 0.;
    #pragma omp parallel for reduction( + : se )
    for (int e = 0; e < m_num_elements; e++) {
        if (lagged_u) {
            se += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (m_Zi[e] - m_prev_Ui[e])).squaredNorm();
        } else {
            se += 0.5 * m_weight_squared[e] * (m_cached_symF[e] - (m_Zi[e] - m_Ui[e])).squaredNorm();
        }
    }
    return se; 
}


double TetElements::primal_residual_sq(Settings::RotAwareness rot_awareness) const {
    double res_sq = 0.;
    if (rot_awareness == Settings::RotAwareness::ENABLED) {
        #pragma omp parallel for reduction ( + : res_sq)
        for (int e = 0; e < m_num_elements; e++) {
            res_sq += m_weight_squared[e] * (m_cached_symF[e] - m_Zi[e]).squaredNorm();
        }
    } else if (rot_awareness == Settings::RotAwareness::DISABLED) {
        #pragma omp parallel for reduction ( + : res_sq)
        for (int e = 0; e < m_num_elements; e++) {
            math::MatX3 Di_x = m_cached_F[e].transpose();
            res_sq += m_weight_squared[e] * (Di_x - m_Zi[e]).squaredNorm();
        }
    } else {
        throw std::runtime_error("Error: Invalid rot_awareness value in TetElemenets::primal_residual_sq");
    }
    return res_sq;
}


void TetElements::update_fix_cache(const math::MatX3 &x_fix) {
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        if (m_num_free_inds[e] < 4) {
            math::MatX3 x_fix_local(4 - m_num_free_inds[e], 3);
            int j = 0;
            for (int i = 0; i < 4; i++) {
                if (m_fix_index_tet[e][i] >= 0) {
                    x_fix_local.row(j) = x_fix.row(m_fix_index_tet[e][i]);
                    j += 1;
                }
            }
            m_Dfix_xfix_local[e] = m_Dfix_local[e] * x_fix_local;
        }
    }
}

void TetElements::update_defo_cache(const math::MatX3 &x_free, bool compute_polar_data) {
    m_xfree_rows = x_free.rows();
    const bool should_compute_polar_data = compute_polar_data;
    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        if (m_num_free_inds[e] == 4) {
            math::Mat4x3 x_free_local;
            x_free_local.row(0) = x_free.row(m_free_index_tet[e][0]);
            x_free_local.row(1) = x_free.row(m_free_index_tet[e][1]);
            x_free_local.row(2) = x_free.row(m_free_index_tet[e][2]);
            x_free_local.row(3) = x_free.row(m_free_index_tet[e][3]);
            m_cached_F[e] = (m_Di_local[e] * x_free_local).transpose();
        } else {
            math::MatX3 x_free_local(m_num_free_inds[e], 3);
            int j = 0;
            for (int i = 0; i < 4; i++) {
                if (m_free_index_tet[e][i] >= 0) {
                    x_free_local.row(j) = x_free.row(m_free_index_tet[e][i]);
                    j += 1;
                }
            }
            m_cached_F[e] = ((m_Dfree_local[e] * x_free_local) + m_Dfix_xfix_local[e]).transpose();
        }
    }

    if (should_compute_polar_data) {
        math::svd(m_cached_F, m_cached_svdsigma, m_cached_svdU, m_cached_svdV);
        #pragma omp parallel for
        for (int e = 0; e < m_num_elements; e++) {
            m_cached_symF[e] = m_cached_svdV[e] * m_cached_svdsigma[e].asDiagonal() * m_cached_svdV[e].transpose();
        }
    }
}

double TetElements::potential_energy_x() const {
    double pe = 0.;

    if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::ARAP) {
        double k = 1.;
        // double k = m_model_settings.elastic_lame().bulk_modulus();
        #pragma omp parallel for reduction ( + : pe)
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x3 RotF;
            math::Mat3x3 SymF;
            math::polar(m_cached_F[e], RotF, SymF, true);
            pe += 0.5 * k * m_volume[e] * (m_cached_F[e] - RotF).squaredNorm();
        }
        return pe;
    } else if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::NH) {
        double lambda = m_model_settings.elastic_lame().lambda();
        double mu = m_model_settings.elastic_lame().mu();
        double J0 = 0.1;
        #pragma omp parallel for reduction ( + : pe)
        for (int e = 0; e < m_num_elements; e++) {
            math::Vec3 sigma = m_cached_svdsigma[e];
	        double I_1 = sigma.squaredNorm();
	        double J = sigma.prod();
            double val = 0.5 * mu * (I_1 - 3.0);
            if (J > J0) {
                double logJ = std::log(J);
                val += -mu*logJ + 0.5*lambda*logJ*logJ;
            } else {
                double fJ = std::log(J0) + ((J - J0) / J0) - (0.5 * std::pow(((J / J0) - 1.0), 2.0));
                val += -mu*fJ + 0.5*lambda*fJ*fJ;
            }
            pe += m_volume[e] * val;
        }
        return pe;
    } else {
        throw std::runtime_error("Error: Invalid elastic model in potential_energy_x");
    }
}


double TetElements::polar_potential_energy() const {
    double pe = 0.;
    if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::ARAP) {
        // double k = m_model_settings.elastic_lame().bulk_modulus();
        double k = 1.;
        #pragma omp parallel for reduction ( + : pe)
        for (int e = 0; e < m_num_elements; e++) {
            pe += 0.5 * k * m_volume[e] * (m_Zi[e] - math::Mat3x3::Identity()).squaredNorm();
        }
    } else if (m_model_settings.elastic_model() == ModelSettings::ElasticModel::NH) {
        double lambda = m_model_settings.elastic_lame().lambda();
        double mu = m_model_settings.elastic_lame().mu();
        double J0 = 0.1;
        #pragma omp parallel for reduction ( + : pe)
        for (int e = 0; e < m_num_elements; e++) {
            math::Mat3x3 S = m_Zi[e].transpose();
            math::Vec3 sigma;
            math::Mat3x3 U;
            math::Mat3x3 V;
            math::svd(S, sigma, U, V, true);             
            double I_1 = sigma.squaredNorm();
	        double J = sigma.prod();
            double val = 0.5 * mu * (I_1 - 3.0);
            if (J > J0) {
                double logJ = std::log(J);
                val += -mu*logJ + 0.5*lambda*logJ*logJ;
            } else {
                double fJ = std::log(J0) + ((J - J0) / J0) - (0.5 * std::pow(((J / J0) - 1.0), 2.0));
                val += -mu*fJ + 0.5*lambda*fJ*fJ;
            }
            pe += m_volume[e] * val;
        }
    } else {
        throw std::runtime_error("Error: Invalid model in polar_potential_energy");
    }
    
    return pe;
}


double TetElements::global_obj_value(const math::MatX3 &x_free) {
    update_defo_cache(x_free);
    double val = 0.;
    #pragma omp parallel for reduction ( + : val )
    for (int e = 0; e < m_num_elements; e++) {
        math::Mat3x3 P = (m_Zi[e] - m_Ui[e]).transpose();
        val += (0.5 * m_weight_squared[e]) * (m_cached_symF[e] - P).squaredNorm();
    }

    return val;    
}


double TetElements::global_obj_grad(const math::MatX3 &x_free, math::MatX3 &gradient) {

    update_defo_cache(x_free);

    gradient = math::MatX3::Zero(x_free.rows(), 3);
    double val = 0.;

    #pragma omp parallel
    {
        math::Mat3x3 P;
        math::Mat3x3 Q;
        math::Mat3x3 K_plus_Kt;
        math::Mat3x3 M;
        math::MatX3 gradient_th = math::MatX3::Zero(m_xfree_rows, 3);
        #pragma omp for reduction ( + : val) nowait
        for (int e = 0; e < m_num_elements; e++) {
            math::Vec3 s = m_cached_svdsigma[e];
            P = (m_Zi[e] - m_Ui[e]).transpose();
            Q = m_cached_svdV[e].transpose() * P * m_cached_svdV[e];

            K_plus_Kt(0, 0) = Q(0, 0) / s[0];
            K_plus_Kt(1, 1) = Q(1, 1) / s[1];
            K_plus_Kt(2, 2) = Q(2, 2) / s[2];
            K_plus_Kt(0, 1) = K_plus_Kt(1, 0) = (Q(0, 1) + Q(1, 0)) / (s[0] + s[1]);
            K_plus_Kt(0, 2) = K_plus_Kt(2, 0) = (Q(0, 2) + Q(2, 0)) / (s[0] + s[2]);
            K_plus_Kt(1, 2) = K_plus_Kt(2, 1) = (Q(1, 2) + Q(2, 1)) / (s[1] + s[2]);

            M = m_cached_F[e] - (m_cached_svdU[e] * m_cached_svdsigma[e].asDiagonal()
                    * K_plus_Kt * m_cached_svdV[e].transpose());

            math::Mat4x3 g = m_Di_local[e].transpose() * ((m_weight_squared[e]) * M.transpose());
            for (int ii = 0; ii < 4; ii++) {
                if (m_free_index_tet[e][ii] != -1) {
                    gradient_th.row(m_free_index_tet[e][ii]) += g.row(ii);
                }
            }

            val += (0.5 * m_weight_squared[e]) * (m_cached_symF[e] - P).squaredNorm();
        }

        #pragma omp critical
        {
            gradient += gradient_th;
        }
    }

    return val;
}



math::MatX3 TetElements::global_obj_grad() const {

    std::vector<math::Mat4x3> gradients(m_num_elements);

    #pragma omp parallel for
    for (int e = 0; e < m_num_elements; e++) {
        math::Vec3 s = m_cached_svdsigma[e];
        math::Mat3x3 Q = m_cached_svdV[e].transpose() * (m_Zi[e] - m_Ui[e]).transpose() * m_cached_svdV[e];

        math::Mat3x3 K_plus_Kt;
        K_plus_Kt(0, 0) = Q(0, 0) / s[0];
        K_plus_Kt(1, 1) = Q(1, 1) / s[1];
        K_plus_Kt(2, 2) = Q(2, 2) / s[2];
        K_plus_Kt(0, 1) = K_plus_Kt(1, 0) = (Q(0, 1) + Q(1, 0)) / (s[0] + s[1]);
        K_plus_Kt(0, 2) = K_plus_Kt(2, 0) = (Q(0, 2) + Q(2, 0)) / (s[0] + s[2]);
        K_plus_Kt(1, 2) = K_plus_Kt(2, 1) = (Q(1, 2) + Q(2, 1)) / (s[1] + s[2]);

        math::Mat3x3 M = m_cached_F[e] - (m_cached_svdU[e] * m_cached_svdsigma[e].asDiagonal()
                * K_plus_Kt * m_cached_svdV[e].transpose());

        gradients[e].noalias() = m_Di_local[e].transpose() * ((m_weight_squared[e]) * M.transpose());
    }

    math::MatX3 gradient = math::MatX3::Zero(m_xfree_rows, 3);

    #pragma omp parallel
    {
        math::MatX3 gradient_th = math::MatX3::Zero(m_xfree_rows, 3);
        #pragma omp for nowait
        for (int e = 0; e < m_num_elements; e++) {
            for (int ii = 0; ii < 4; ii++) {
                if (m_free_index_tet[e][ii] != -1) {
                    gradient_th.row(m_free_index_tet[e][ii]) += gradients[e].row(ii);
                }
            }
        }

        #pragma omp critical
        {
            gradient += gradient_th;
        }
    }

    return gradient;

}

}  // namespace wrapd
