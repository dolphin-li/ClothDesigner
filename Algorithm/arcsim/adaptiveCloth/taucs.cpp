/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "taucs.hpp"
#include "timer.hpp"
#include <cstdlib>
#include <iostream>

#define USE_EIGEN_INSTEADOF_TAUCS

using namespace std;

namespace arcsim
{
#ifdef USE_EIGEN_INSTEADOF_TAUCS
#include <eigen\Sparse>
#include <eigen\Dense>

	void sparse_to_eigen(const SpMat<double> &As, Eigen::SparseMatrix<double>& A)
	{
		vector<Eigen::Triplet<double>> cooSys;
		// assumption: A is square and symmetric
		A.resize(As.n, As.n);
		for (int i = 0; i < As.n; i++)
		{
			for (int k = 0; k < As.rows[i].indices.size(); k++)
				cooSys.push_back(Eigen::Triplet<double>(As.rows[i].indices[k], i, As.rows[i].entries[k]));
		}
		if (cooSys.size())
			A.setFromTriplets(cooSys.begin(), cooSys.end());
	}	
	template <int m> void sparse_to_eigen(const SpMat< Mat<m, m> > &As, Eigen::SparseMatrix<double>& A)
	{
		vector<Eigen::Triplet<double>> cooSys;
		// assumption: A is square and symmetric
		A.resize(As.n * m, As.n * m);
		for (int i = 0; i < As.n; i++)
		{
			for (int k = 0; k < m; k++)
			{
				for (int jj = 0; jj < As.rows[i].indices.size(); jj++)
				{
					int j = As.rows[i].indices[jj];
					const Mat<m, m> &Aij = As.rows[i].entries[jj];
					for (int l = 0; l < m; l++)
						cooSys.push_back(Eigen::Triplet<double>(j*m + l, i*m + k, Aij(k, l)));
				}
			}
		}
		if (cooSys.size())
			A.setFromTriplets(cooSys.begin(), cooSys.end());
	}
#else
	extern "C" {
#include "taucs\include\taucs.h"
		int taucs_linsolve(taucs_ccs_matrix* A, // input matrix
			void** factorization, // an approximate inverse
			int nrhs, // number of right-hand sides
			void* X, // unknowns
			void* B, // right-hand sides
			char* options[], // options (what to do and how)
			void* arguments[]); // option arguments
	}

	ostream &operator<< (ostream &out, taucs_ccs_matrix *A)
	{
		out << "n: " << A->n << endl;
		out << "m: " << A->m << endl;
		out << "flags: " << A->flags << endl;
		out << "colptr: ";
		for (int i = 0; i <= A->n; i++)
			out << (i == 0 ? "" : ", ") << A->colptr[i];
		out << endl;
		out << "rowind: ";
		for (int j = 0; j <= A->colptr[A->n]; j++)
			out << (j == 0 ? "" : ", ") << A->rowind[j];
		out << endl;
		out << "values.d: ";
		for (int j = 0; j <= A->colptr[A->n]; j++)
			out << (j == 0 ? "" : ", ") << A->values.d[j];
		out << endl;
		return out;
	}

	taucs_ccs_matrix *sparse_to_taucs(const SpMat<double> &As)
	{
		// assumption: A is square and symmetric
		int n = As.n;
		int nnz = 0;
		for (int i = 0; i < n; i++)
		{
			for (int k = 0; k < As.rows[i].indices.size(); k++)
			{
				int j = As.rows[i].indices[k];
				if (j < i)
					continue;
				nnz++;
			}
		}
		taucs_ccs_matrix *At = taucs_ccs_create(n, n, nnz, TAUCS_DOUBLE | TAUCS_SYMMETRIC | TAUCS_LOWER);
		int pos = 0;
		for (int i = 0; i < n; i++)
		{
			At->colptr[i] = pos;
			for (int k = 0; k < As.rows[i].indices.size(); k++)
			{
				int j = As.rows[i].indices[k];
				if (j < i)
					continue;
				At->rowind[pos] = j;
				At->values.d[pos] = As.rows[i].entries[k];
				pos++;
			}
		}
		At->colptr[n] = pos;
		return At;
	}	
	template <int m> taucs_ccs_matrix *sparse_to_taucs(const SpMat< Mat<m, m> > &As)
	{
		// assumption: A is square and symmetric
		int n = As.n;
		int nnz = 0;
		for (int i = 0; i < n; i++)
		{
			for (int jj = 0; jj < As.rows[i].indices.size(); jj++)
			{
				int j = As.rows[i].indices[jj];
				if (j < i)
					continue;
				nnz += (j == i) ? m*(m + 1) / 2 : m*m;
			}
		}
		taucs_ccs_matrix *At = taucs_ccs_create
			(n*m, n*m, nnz, TAUCS_DOUBLE | TAUCS_SYMMETRIC | TAUCS_LOWER);
		int pos = 0;
		for (int i = 0; i < n; i++)
		{
			for (int k = 0; k < m; k++)
			{
				At->colptr[i*m + k] = pos;
				for (int jj = 0; jj < As.rows[i].indices.size(); jj++)
				{
					int j = As.rows[i].indices[jj];
					if (j < i)
						continue;
					const Mat<m, m> &Aij = As.rows[i].entries[jj];
					for (int l = (i == j) ? k : 0; l < m; l++)
					{
						At->rowind[pos] = j*m + l;
						At->values.d[pos] = Aij(k, l);
						pos++;
					}
				}
			}
		}
		At->colptr[n*m] = pos;
		return At;
	}
#endif


	vector<double> taucs_linear_solve(const SpMat<double> &A, const vector<double> &b)
	{
#ifdef USE_EIGEN_INSTEADOF_TAUCS
		typedef Eigen::SparseMatrix<double> SpMat;
		typedef Eigen::MatrixXd DMat;
		typedef Eigen::VectorXd DVec;
		typedef Eigen::SimplicialCholesky<SpMat> Solver;
		SpMat eA;
		DVec eb(b.size());
		for (size_t i = 0; i < b.size(); i++)
			eb[i] = b[i];
		sparse_to_eigen(A, eA);
		Solver solver(eA);
		DVec ex = solver.solve(eb);
		vector<double> x(ex.size());
		for(size_t i=0; i<x.size(); i++)
			x[i] = ex[i];
		return x;
#else
		// taucs_logfile("stdout");
		taucs_ccs_matrix *Ataucs = sparse_to_taucs(A);
		vector<double> x(b.size());
		char *options[] = { (char*)"taucs.factor.LLT=true", NULL };
		//debug_save_spmat(A);
		int retval = taucs_linsolve(Ataucs, NULL, 1, &x[0], (double*)&b[0], options, NULL);
		if (retval != TAUCS_SUCCESS)
		{
			cerr << "Error: TAUCS failed with return value " << retval << endl;
			exit(EXIT_FAILURE);
		}
		taucs_ccs_free(Ataucs);
		return x;
#endif
	}

	template <int m> vector< Vec<m> > taucs_linear_solve
		(const SpMat< Mat<m, m> > &A, const vector< Vec<m> > &b)
	{
#ifdef USE_EIGEN_INSTEADOF_TAUCS
		typedef Eigen::SparseMatrix<double> SpMat;
		typedef Eigen::MatrixXd DMat;
		typedef Eigen::VectorXd DVec;
		typedef Eigen::SimplicialCholesky<SpMat> Solver;
		SpMat eA;
		DVec eb(b.size()*m);
		for (size_t i = 0; i < b.size(); i++)
		{
			for (int k = 0; k < m; k++)
				eb[i*m + k] = b[i][k];
		}
		sparse_to_eigen(A, eA);
		Solver solver(eA);
		DVec ex = solver.solve(eb);
		vector<Vec<m>> x(ex.size()/m);
		for (size_t i = 0; i < x.size(); i++)
		{
			for (int k = 0; k < m; k++)
				x[i][k] = ex[i*m + k];
		}
		return x;
#else
		// taucs_logfile("stdout");
		taucs_ccs_matrix *Ataucs = sparse_to_taucs(A);
		vector< Vec<m> > x(b.size());
		char *options[] = { (char*)"taucs.factor.LLT=true", NULL };
		int retval = taucs_linsolve(Ataucs, NULL, 1, &x[0], (double*)&b[0], options, NULL);
		if (retval != TAUCS_SUCCESS)
		{
			cerr << "Error: TAUCS failed with return value " << retval << endl;
			exit(EXIT_FAILURE);
		}
		taucs_ccs_free(Ataucs);
		return x;
#endif
	}

	template vector<Vec3> taucs_linear_solve(const SpMat<Mat3x3> &A,
		const vector<Vec3> &b);
}