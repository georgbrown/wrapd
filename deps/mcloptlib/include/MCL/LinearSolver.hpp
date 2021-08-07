
#ifndef ADMM_LINEARSOLVER_H_
#define ADMM_LINEARSOLVER_H_ 1

#include <Eigen/SparseCholesky>
#include <Eigen/Dense>

namespace mcl {
namespace optlib {

#ifdef USE_PARDISO

// Pariso-project functions
extern "C" {
    void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
    void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                    double *, int    *,    int *, int *,   int *, int *,
                    int *, double *, double *, int *, double *);
    void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
    void pardiso_chkvec     (int *, int *, double *, int *);
    void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);
    void pardiso_residual(int *, int *, double *, int *, int *, double *, double *, double *, double *, double *);
}

class PardisoSolver {
public:
    enum {
        SPD = 2,
        SI = -2
    };

	PardisoSolver() :
		mtype(SPD),
		nrhs(1), // number of right hand side
		msglvl(0), // 0=quiet, 1=error checking
		n(0), // dof
		maxfct(1), // max number of factorizations
		mnum(1), // which factorization to use
		solver(0), // sparse direct solver
		m_initialized(false)
		{}

	PardisoSolver(Eigen::SparseMatrix<double, Eigen::RowMajor> &A) : PardisoSolver() { compute(A); }

    ~PardisoSolver(){ 
		if (m_initialized) {
			pardisoRelease();
		}
	}

	void compute(Eigen::SparseMatrix<double, Eigen::RowMajor> &A) {
		    m_info = Eigen::InvalidInput;

		using namespace Eigen;

		int dof = A.rows();
		m_x.resize(dof);
		if (dof != A.cols() || dof == 0) {
			pardisoErrorCode(-1,"compute",true);
			return;
		}

		// TODO just create CSR from column major A
		// SparseMatrix<double,RowMajor> A = A_;//.triangularView<Upper>();

		n = A.rows();
		int nnz = A.nonZeros();
		int error = 0;

		m_perm = VectorXi::Zero(n);
		m_rows.clear();
		m_rows.reserve(n+1);
		m_cols.clear();
		m_cols.reserve(nnz);
		m_vals.clear();
		m_vals.reserve(nnz);
		VectorXd x_dummy = VectorXd::Zero(n);

		// Convert from row matrix to CSR and Fortran (+1) index
		m_rows.emplace_back(1);
		for (int r=0, v_idx=0; r<n; ++r) {
			m_perm[r] = r+1;
			SparseMatrix<double,RowMajor>::InnerIterator it(A, r);
			for( ; it; ++it ){
				int col = it.col();
				if (col >= r) { // upper triangular
					m_vals.emplace_back(it.value());
					m_cols.emplace_back(col+1);
					++v_idx;
				}
			} // end loop nonzeros for col c
			m_rows.emplace_back(v_idx+1); // Update row index
		} // end loop cols

		// Check input matrix
		if (msglvl>0) {
			pardiso_chkmatrix(&mtype, &n, m_vals.data(), m_rows.data(), m_cols.data(), &error);
			pardisoErrorCode(error,"compute",true);
			if (error != 0) {
				return;
			}
		}

		int n_threads = 0;
		char *ont = getenv("OMP_NUM_THREADS");
		if (ont != NULL) { sscanf(ont, "%d", &n_threads); }
		if (n_threads <= 0) {
			std::string err = "**Error in PardisoSolver::compute: Must set env OMP_NUM_THREADS";
			throw std::runtime_error(err.c_str());
		}

		// Tests for pardiso-project license and inits solver
		// iparm[2] must be OMP_NUM_THREADS
		iparm[2] = n_threads;
		iparm[0] = 0; // Sets parameters to default (except iparm 2)
		pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
		pardisoErrorCode(error,"compute",true);
		if (error != 0) {
			m_info =  Eigen::InvalidInput;
			return;
		}

		iparm[0] = 1; // no longer use default params
		iparm[33] = 1; // identical perf on multicore
		iparm[29] = 20; // supernodes

		int phase = 11; // Symbolic factorization
		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
				&n, m_vals.data(), m_rows.data(), m_cols.data(), m_perm.data(), &nrhs,
				iparm, &msglvl, NULL, NULL, &error, dparm);
		pardisoErrorCode(error,"compute",true);
		if (error != 0) {
			m_info =  Eigen::NumericalIssue;
			return;
		}

		phase = 22; // Numerical factorization
		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
				&n, m_vals.data(), m_rows.data(), m_cols.data(), m_perm.data(), &nrhs,
				iparm, &msglvl, NULL, NULL, &error, dparm);
		pardisoErrorCode(error,"compute",true);
		if (error != 0) {
			m_info = Eigen::NumericalIssue;
			return;
		}

		m_info = Eigen::Success;
		m_initialized = true;
	}

	void factorize(Eigen::SparseMatrix<double, Eigen::RowMajor> &A) {
		using namespace Eigen;
		// SparseMatrix<double,RowMajor> A = A_;

		// Converge row matrix to CSR and Fortran (+1) index
		m_rows[0]=1;
		for (int r=0, v_idx=0; r<n; ++r) {
			SparseMatrix<double,RowMajor>::InnerIterator it(A, r);
			for( ; it; ++it ){
				int col = it.col();
				if (col >= r) { // upper triangular
					m_vals[v_idx] = it.value();
					m_cols[v_idx] = col+1;
					++v_idx;
				}
			} // end loop nonzeros for col c
			m_rows[r+1] = v_idx+1;
		} // end loop cols

		int error = 0;
		int phase = 22; // Numerical factorization
		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
				&n, m_vals.data(), m_rows.data(), m_cols.data(), m_perm.data(), &nrhs,
				iparm, &msglvl, NULL, NULL, &error, dparm);
		pardisoErrorCode(error,"factorize",true);
		if (error != 0) {
			m_info =  Eigen::NumericalIssue;
		}		
	}

	Eigen::ComputationInfo info(){ return m_info; }

	Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
		using namespace Eigen;
		if (m_info != Eigen::Success) {
			throw std::runtime_error("**PardisoSolver::solve Error: Cannot solve, no factorization\n");
			return VectorXd::Zero(rhs.rows());
		}
		iparm[8] = 1; // ???
		int phase = 33;
		int error = 0;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
				&n, m_vals.data(), m_rows.data(), m_cols.data(), m_perm.data(), &nrhs,
				iparm, &msglvl, (double*)rhs.data(), m_x.data(), &error, dparm);
		pardisoErrorCode(error,"solve",true);

		return m_x;
	}

    // Set matrix type. Default SPD
    int &type() { return mtype; }

	void pardisoErrorCode(int error, std::string call, bool ex=false) const {
		std::string err = "";
		switch(error) {
		case 0: // success
			break;
		case -1: err =  "\n**Error in PardisoSolver::%s: input inconsistent\n" + call;
			break;
		case -2: err =  "\n**Error in PardisoSolver::%s: not enough memory\n" + call;
			break;
		case -3: err =  "\n**Error in PardisoSolver::%s: reordering problem\n" + call;
			break;
		case -4: err =  "\n**Error in PardisoSolver::%s: zero pivot, numerical factorization or iterative refinement problem\n" + call;
			break;
		case -5: err =  "\n**Error in PardisoSolver::%s: unclassified (internal) error\n" + call;
			break;
		case -6: err =  "\n**Error in PardisoSolver::%s: pre-ordering failed (matrix types 11, 13 only)\n" + call;
			break;
		case -7: err =  "\n**Error in PardisoSolver::%s: diagonal matrix is singular\n" + call;
			break;
		case -8: err =  "\n**Error in PardisoSolver::%s: 32-bit integer overflow problem\n" + call;
			break;
		case -10: err =  "\n**Error in PardisoSolver::%s: No license file pardiso.lic found.\n" + call;
			break;
		case -11: err =  "\n**Error in PardisoSolver::%s: License is expired.\n" + call;
			break;
		case -12: err =  "\n**Error in PardisoSolver::%s: Wrong username or hostname.\n" + call;
			break;
		case -100: err =  "\n**Error in PardisoSolver::%s: Reached maximum number of Krylov-subspace iteration in iterative solver.\n" + call;
			break;
		case -101: err =  "\n**Error in PardisoSolver::%s: No sufficient convergence in Krylov-subspace iteration within 25 iterations.\n" + call;
			break;
		case -102: err =  "\n**Error in PardisoSolver::%s: Error in Krylov-subspace iteration.\n" + call;
			break;
		case -103: err =  "\n**Error in PardisoSolver::%s: Break-Down in Krylov-subspace iteration\n" + call;
			break;
		default: err =  "\n**Unknown Error in PardisoSolver::%s" + call;
			break;
		}
		if (err.size()>0) {
			if (ex) { throw std::runtime_error(err.c_str()); }
			else { printf("%s",err.c_str()); }
		}
    }		

	void pardisoRelease() {
		if (m_info != Eigen::Success) { return; }
		int phase = -1;
		int error = 0;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase,
			&n, m_vals.data(), m_rows.data(), m_cols.data(), m_perm.data(), &nrhs,
			iparm, &msglvl, NULL, NULL, &error, dparm);
		pardisoErrorCode(error,"release",false);
	}

protected:
	int mtype;
	int nrhs;
	int msglvl;
	int n;
	int maxfct, mnum, solver;
	long int pt[64]; // int[64] if 32bit, long int[64] if 64bit, void *pt[64] works for both
	int iparm[64];
	double dparm[64];
	std::vector<int> m_rows;
	std::vector<int> m_cols;
	std::vector<double> m_vals;
	Eigen::VectorXi m_perm;
	Eigen::VectorXd m_x;  // solution from previous solve
	Eigen::ComputationInfo m_info;
	bool m_initialized;

}; // end class PardisoSolver

#endif

// Wrapper for whatever linear solver I decide to go with in the future
class LinearSolver {
protected:
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> cholesky;

public:

    LinearSolver() {}
    LinearSolver(const Eigen::SparseMatrix<double> &A) : LinearSolver() { compute(A); }

	void compute(const Eigen::SparseMatrix<double> &A){
		cholesky.analyzePattern(A);
		cholesky.factorize(A);
	}
	void factorize(const Eigen::SparseMatrix<double> &A) { cholesky.factorize(A); }

	Eigen::ComputationInfo info(){ return cholesky.info(); }

	Eigen::VectorXd solve(const Eigen::VectorXd &rhs){ return cholesky.solve(rhs); }

}; // end class linear solver

} // ns optlib
} // ns mcl

#endif
