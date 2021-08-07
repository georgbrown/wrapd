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

#ifndef SRC_MATH_HPP_
#define SRC_MATH_HPP_

#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace wrapd {
namespace math {
    
using VecX = Eigen::VectorXd;
using RowVecX = Eigen::RowVectorXd;
using VecXi = Eigen::VectorXi;
using MatX = Eigen::MatrixXd;
using MatXi = Eigen::MatrixXi;
using MatX2 = Eigen::MatrixX2d;
using Mat2X = Eigen::Matrix2Xd;
using MatX3 = Eigen::MatrixX3d;
using Mat3X = Eigen::Matrix3Xd;
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using DiagMat = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using RowVec3 = Eigen::RowVector3d;
using RowVec4 = Eigen::RowVector4d;

using Triplet = Eigen::Triplet<double>;
using Triplets = std::vector<Triplet>;

using Doubles = std::vector<double>;

using Cholesky = Eigen::SimplicialLDLT< Eigen::SparseMatrix<double>, Eigen::Lower>;

using ConjugateGradient = Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>;

using AlignedBox = Eigen::AlignedBox<double, 3>;

// Singular value decomposition
template <typename TYPE>
using JacobiSVD = Eigen::JacobiSVD<TYPE>;
static constexpr int ComputeFullU = Eigen::ComputeFullU;
static constexpr int ComputeFullV = Eigen::ComputeFullV;

// Mapping between vectors and matrices
template <typename TYPE>
using Map = Eigen::Map<TYPE>;

// Vectors of doubles
template <int DIM>
using Vec = Eigen::Matrix<double, DIM, 1>;
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;
using Vec5 = Vec<5>;
using Vec6 = Vec<6>;
using Vec7 = Vec<7>;
using Vec8 = Vec<8>;
using Vec9 = Vec<9>;
using Vec10 = Vec<10>;
using Vec11 = Vec<11>;
using Vec12 = Vec<12>;

// Vectors of floats
template <int DIM>
using Vecf = Eigen::Matrix<float, DIM, 1>;
using Vec2f = Vecf<2>;
using Vec3f = Vecf<3>;
using Vec4f = Vecf<4>;

// Vectors of integers
template <int DIM>
using Veci = Eigen::Matrix<int, DIM, 1>;
using Vec1i = Veci<1>;
using Vec2i = Veci<2>;
using Vec3i = Veci<3>;
using Vec4i = Veci<4>;

template <int ROWS, int COLS>
using Mat = Eigen::Matrix<double, ROWS, COLS>;

using Mat1x3 = Mat<1, 3>;
using Mat2x2 = Mat<2, 2>;
using Mat2x3 = Mat<2, 3>;
using Mat3x2 = Mat<3, 2>;
using Mat3x3 = Mat<3, 3>;
using Mat3x4 = Mat<3, 4>;
using Mat3x9 = Mat<3, 9>;
using Mat3x12 = Mat<3, 12>;
using Mat4x3 = Mat<4, 3>;
using Mat4x4 = Mat<4, 4>;
using Mat4x6 = Mat<4, 6>;
using Mat4x9 = Mat<4, 9>;
using Mat6x4 = Mat<6, 4>;
using Mat6x6 = Mat<6, 6>;
using Mat6x9 = Mat<6, 9>;
using Mat9x4 = Mat<9, 4>;
using Mat9x6 = Mat<9, 6>;
using Mat9x9 = Mat<9, 9>;
using Mat9x12 = Mat<9, 12>;
using Mat12x3 = Mat<12, 3>;
using Mat12x9 = Mat<12, 9>;
using Mat12x12 = Mat<12, 12>;

inline double clamp(const double minval, const double val, const double maxval) {
    double retval = val;
    if (val < minval) {
        retval = minval;
    }
    if (val > maxval) {
        retval = maxval;
    }
    return retval;
}

inline void svd(const math::Mat3x3& F, math::Vec3& sigma, math::Mat3x3& U, math::Mat3x3& V, bool identify_flips = true) {
    
    Eigen::JacobiSVD<math::Mat3x3, Eigen::FullPivHouseholderQRPreconditioner> svd(F, math::ComputeFullU | math::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (identify_flips) {
        math::Mat3x3 J = math::Mat3x3::Identity();
        J(2, 2) = -1.0;
        if (U.determinant() < 0.) {
            U = U * J;
            sigma[2] = -sigma[2];
        }
        if (V.determinant() < 0.0) {
            math::Mat3x3 Vt = V.transpose();
            Vt = J * Vt;
            V = Vt.transpose();
            sigma[2] = -sigma[2];
        }
    }        
}

inline void svd(const std::vector<math::Mat3x3> &A, std::vector<math::Vec3> &sigma,
        std::vector<math::Mat3x3> &U, std::vector<math::Mat3x3> &V) {

    const int num_elements = A.size();

    using EigenSVD = Eigen::JacobiSVD<math::Mat3x3, Eigen::FullPivHouseholderQRPreconditioner>;

    #pragma omp parallel
    {
        EigenSVD svd(3, 3, math::ComputeFullU | math::ComputeFullV);
        #pragma omp for
        for (int e = 0; e < num_elements; e++) {
            svd.compute(A[e]);
            sigma[e] = svd.singularValues();
            U[e] = svd.matrixU();
            V[e] = svd.matrixV();
            math::Mat3x3 J = math::Mat3x3::Identity();
            J(2, 2) = -1.0;
            if (U[e].determinant() < 0.) {
                U[e] = U[e] * J;
                sigma[e][2] = -sigma[e][2];
            }
            if (V[e].determinant() < 0.0) {
                math::Mat3x3 Vt = V[e].transpose();
                Vt = J * Vt;
                V[e] = Vt.transpose();
                sigma[e][2] = -sigma[e][2];
            }
        }            
    }
}

inline void svd(const math::Mat2x2 &A, math::Vec2 &sigma, math::Mat2x2 &U,
        math::Mat2x2 &V, bool identify_flips = true) {
    Eigen::JacobiSVD<math::Mat2x2, Eigen::FullPivHouseholderQRPreconditioner> svd(A, math::ComputeFullU | math::ComputeFullV);
    sigma = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    if (identify_flips) {
        math::Mat2x2 J = math::Mat2x2::Identity();
        J(1, 1) = -1.0;
        if (U.determinant() < 0.) {
            U = U * J;
            sigma[1] = -sigma[1];
        }
        if (V.determinant() < 0.0) {
            math::Mat2x2 Vt = V.transpose();
            Vt = J * Vt;
            V = Vt.transpose();
            sigma[1] = -sigma[1];
        }
    }
}

inline void polar(const math::Mat3x3& F, math::Mat3x3& R, math::Mat3x3& S, bool identify_flips = true) {
    math::Vec3 sigma;
    math::Mat3x3 U;
    math::Mat3x3 V;
    svd(F, sigma, U, V, identify_flips);
    R = U * V.transpose();
    S = V * sigma.asDiagonal() * V.transpose();
}

inline math::Mat3x3 rot(const math::Mat3x3& A) {
    math::Mat3x3 R;
    math::Mat3x3 S;
    polar(A, R, S, true);
    return R;
}

inline math::Mat3x3 sym(const math::Mat3x3& A) {
    math::Mat3x3 R;
    math::Mat3x3 S;
    polar(A, R, S, true);
    return S;
}

// Project a symmetric real matrix to the nearest SPD matrix
// Source: https://github.com/liminchen/OptCuts/blob/master/src/Utils/IglUtils.hpp#L72
template<typename Scalar, int size>
inline void makePD(Eigen::Matrix<Scalar, size, size>& symMtr) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if(eigenSolver.eigenvalues()[0] >= 0.0) {
        return;
    }
    Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
    int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
    int i = 0;
    for(; i < rows; i++) {
        if(D.diagonal()[i] < 0.0) {
            D.diagonal()[i] = 0.0;
        }
        else {
            break;
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

}  // namespace math
}  // namespace wrapd

#endif  // SRC_MATH_HPP_
