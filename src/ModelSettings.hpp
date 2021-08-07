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

#ifndef SRC_MODELSETTINGS_HPP_
#define SRC_MODELSETTINGS_HPP_

#include "Lame.hpp"
#include "Math.hpp"

namespace wrapd {

class ModelSettings {
 public:
    ModelSettings() :
        m_elastic_model(ElasticModel::NH),
        m_elastic_lame(Lame::preset(0.45)),
        m_elastic_beta_static(1.0),
        m_elastic_beta_min(0.1),
        m_elastic_beta_max(10.0),
        m_elastic_prox_ls_method(ProxLSMethod::BacktrackingCubic),
        m_elastic_prox_iters(20) {
    }

    enum class ElasticModel {
        ARAP,
        NH,
    };

    enum class ProxLSMethod {
        None,
        MoreThuente,
        Backtracking,
        BacktrackingCubic,
        WeakWolfeBisection,
    };

    ElasticModel elastic_model() const { return m_elastic_model; }
    void elastic_model(std::string val) {
        if (val == "arap") {
            m_elastic_model = ElasticModel::ARAP;
        } else if (val == "nh") {
            m_elastic_model = ElasticModel::NH;
        } else {
            throw std::runtime_error("Error: Invalid input for -elastic_model");
        }
    }

    void poisson(double val) {
        m_elastic_lame = Lame::preset(val);
    }

    const Lame& elastic_lame() const { return m_elastic_lame; }

    double elastic_beta_static() const { return m_elastic_beta_static; }
    void elastic_beta_static(double val) { m_elastic_beta_static = val; }

    double elastic_beta_min() const { return m_elastic_beta_min; }
    void elastic_beta_min(double val) { m_elastic_beta_min = val; }

    double elastic_beta_max() const { return m_elastic_beta_max; }
    void elastic_beta_max(double val) { m_elastic_beta_max = val; }

    ProxLSMethod elastic_prox_ls_method() const { return m_elastic_prox_ls_method; }
    int elastic_prox_iters() const { return m_elastic_prox_iters; }

 private:
    ElasticModel m_elastic_model;
    Lame m_elastic_lame;
    double m_elastic_beta_static;
    double m_elastic_beta_min;
    double m_elastic_beta_max;

    ProxLSMethod m_elastic_prox_ls_method;
    int m_elastic_prox_iters;
    
};
}  // namespace wrapd

#endif  // SRC_MODELSETTINGS_HPP_
