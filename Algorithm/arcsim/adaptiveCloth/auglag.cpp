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

#include "optimization.hpp"

#include "alglib/optimization.h"
#include <omp.h>
#include <vector>
using namespace std;
using namespace alglib;

namespace arcsim
{
	static const NLConOpt *problem;
	static vector<double> lambda;
	static double mu;

	static void auglag_value_and_grad(const real_1d_array &x, double &value,
		real_1d_array &grad, void *ptr = NULL);

	static void multiplier_update(const real_1d_array &x);

	void augmented_lagrangian_method(const NLConOpt &problem, OptOptions opt,
		bool verbose)
	{
		arcsim::problem = &problem;
		arcsim::lambda = vector<double>(arcsim::problem->ncon, 0);
		arcsim::mu = 1e3;
		real_1d_array x;
		x.setlength(arcsim::problem->nvar);
		arcsim::problem->initialize(&x[0]);
		mincgstate state;
		mincgreport rep;
		mincgcreate(x, state);
		const int max_total_iter = opt.max_iter(),
			max_sub_iter = sqrt(max_total_iter);
		int iter = 0;
		while (iter < max_total_iter)
		{
			int max_iter = std::min(max_sub_iter, max_total_iter - iter);
			mincgsetcond(state, opt.eps_g(), opt.eps_f(), opt.eps_x(), max_iter);
			if (iter > 0)
				mincgrestartfrom(state, x);
			mincgsuggeststep(state, 1e-3*arcsim::problem->nvar);
			mincgoptimize(state, auglag_value_and_grad);
			mincgresults(state, x, rep);
			multiplier_update(x);
			if (verbose)
				cout << rep.iterationscount << " iterations" << endl;
			if (rep.iterationscount == 0)
				break;
			iter += rep.iterationscount;
		}
		arcsim::problem->finalize(&x[0]);
	}

	static void add(real_1d_array &x, const vector<double> &y)
	{
		for (int i = 0; i < y.size(); i++)
			x[i] += y[i];
	}

	inline double clamp_violation(double x, int sign)
	{
		return (sign<0) ? std::max(x, 0.) : (sign>0) ? std::min(x, 0.) : x;
	}

	static void auglag_value_and_grad(const real_1d_array &x, double &value,
		real_1d_array &grad, void *ptr)
	{
		arcsim::problem->precompute(&x[0]);
		value = arcsim::problem->objective(&x[0]);
		arcsim::problem->obj_grad(&x[0], &grad[0]);
		static const int nthreads = omp_get_max_threads();
		static double *values = new double[nthreads];
		static vector<double> *grads = new vector<double>[nthreads];
		for (int t = 0; t < nthreads; t++)
		{
			values[t] = 0;
			grads[t].assign(arcsim::problem->nvar, 0);
		}
#pragma omp parallel for
		for (int j = 0; j < arcsim::problem->ncon; j++)
		{
			int t = omp_get_thread_num();
			int sign;
			double gj = arcsim::problem->constraint(&x[0], j, sign);
			double cj = clamp_violation(gj + arcsim::lambda[j] / arcsim::mu, sign);
			if (cj != 0)
			{
				values[t] += arcsim::mu / 2 * sq(cj);
				arcsim::problem->con_grad(&x[0], j, arcsim::mu*cj, &grads[t][0]);
			}
		}
		for (int t = 0; t < nthreads; t++)
			value += values[t];
#pragma omp parallel for
		for (int i = 0; i < arcsim::problem->nvar; i++)
		for (int t = 0; t < nthreads; t++)
			grad[i] += grads[t][i];
	}

	static void multiplier_update(const real_1d_array &x)
	{
		arcsim::problem->precompute(&x[0]);
#pragma omp parallel for
		for (int j = 0; j < arcsim::problem->ncon; j++)
		{
			int sign;
			double gj = arcsim::problem->constraint(&x[0], j, sign);
			arcsim::lambda[j] = clamp_violation(arcsim::lambda[j] + arcsim::mu*gj, sign);
		}
	}
}