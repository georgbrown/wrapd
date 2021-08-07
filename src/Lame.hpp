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

#ifndef SRC_LAME_HPP_
#define SRC_LAME_HPP_

#include <memory>
#include <iostream>


namespace wrapd {

//
// Lame constants
//
class Lame {
 public:
    static Lame preset(double poisson) { return Lame(10'000'000, poisson); }

    void mu(double val) { m_mu = val; }
    double mu() const { return m_mu; }

    void lambda(double val) { m_lambda = val; }
    double lambda() const { return m_lambda; }

    // double bulk_modulus() const { return m_lambda + ((2.0 / 3.0) * m_mu); }

    // k: Youngs (Pa), measure of stretch
    // v: Poisson, measure of incompressibility
    Lame(double k, double v) :
        m_mu(k / (2.0 * (1.0 + v))),
        m_lambda(k * v / ((1.0 + v) * (1.0 - 2.0 * v))) {
    }

 private:
    double m_mu;
    double m_lambda;
};

}  // namespace wrapd

#endif  // SRC_LAME_HPP_
