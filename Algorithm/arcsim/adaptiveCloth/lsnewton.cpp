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

#include "taucs.hpp"
#include "util.hpp"

using namespace std;

namespace arcsim
{

	static bool verbose;

	double line_search(const vector<double> &x0, const vector<double> &p,
		const NLOpt &problem, double f, const vector<double> &g);

	static void add(vector<double> &v, double a, const vector<double> &x,
		double b, const vector<double> &y); //v=ax+by
	static void scalar_mult(vector<double> &v, double a, const vector<double> &x); //v=ax
	static double dot(const vector<double> &x, const vector<double> &y);
	static double norm(const vector<double> &x) { return sqrt(dot(x, x)); }

	void line_search_newtons_method(const NLOpt &problem, OptOptions opt,
		bool verbose)
	{
		arcsim::verbose = verbose;
		int n = problem.nvar;
		vector<double> x(n), g(n);
		SpMat<double> H(n, n);
		problem.initialize(&x[0]);
		double f_old = infinity;
		int iter;
		for (iter = 0; iter < opt.max_iter(); iter++)
		{
			problem.precompute(&x[0]);
			double f = problem.objective(&x[0]);
			if (verbose)
				REPORT(f);
			if (f_old - f < opt.eps_f())
				break;
			f_old = f;
			problem.gradient(&x[0], &g[0]);
			if (verbose)
				REPORT(norm(g));
			if (norm(g) < opt.eps_g())
				break;
			if (!problem.hessian(&x[0], H))
			{
				cerr << "Can't run Newton's method if Hessian of objective "
					<< "is not available!" << endl;
				exit(1);
			}
			vector<double> p = taucs_linear_solve(H, g);
			if (verbose)
				REPORT(norm(p));
			scalar_mult(p, -1, p);
			double a = line_search(x, p, problem, f, g);
			add(x, 1, x, a, p);
			if (a*norm(p) < opt.eps_x())
				break;
		}
		if (verbose)
			REPORT(iter);
		problem.finalize(&x[0]);
	}

	inline double cb(double x) { return x*x*x; }

	double line_search(const vector<double> &x0, const vector<double> &p,
		const NLOpt &problem, double f0, const vector<double> &g)
	{
		double c = 1e-3; // sufficient decrease parameter
		double a = 1;
		int n = problem.nvar;
		vector<double> x(n);
		double g0 = dot(g, p);
		if (arcsim::verbose)
			REPORT(g0);
		if (abs(g0) < 1e-12)
			return 0;
		double a_prev = 0;
		double f_prev = f0;
		while (true)
		{
			add(x, 1, x0, a, p);
			// problem.precompute(&x[0]);
			double f = problem.objective(&x[0]);
			if (arcsim::verbose)
			{
				REPORT(a);
				REPORT(f);
			}
			if (f <= f0 + c*a*g0)
				break;
			double a_next;
			if (a_prev == 0)
				a_next = -g0*sq(a) / (2 * (f - f0 - g0*a)); // minimize quadratic fit
			else
			{
				// minimize cubic fit to f0, g0, f_prev, f
				Vec2 b = Mat2x2(Vec2(sq(a_prev), -cb(a_prev)),
					Vec2(-sq(a), cb(a)))
					* Vec2(f - f0 - g0*a, f_prev - f0 - g0*a_prev)
					/ (sq(a)*sq(a_prev)*(a - a_prev));
				double a_sol[2];
				solve_quadratic(3 * b[0], 2 * b[1], g0, a_sol);
				a_next = (a_sol[0] > 0) ? a_sol[0] : a_sol[1];
			}
			if (a_next < a*0.1 || a_next > a*0.9)
				a_next = a / 2;
			a_prev = a;
			f_prev = f;
			a = a_next;
		}
		return a;
	}

	static void add(vector<double> &v, double a, const vector<double> &x,
		double b, const vector<double> &y)
	{
		int n = x.size();
		v.resize(n);
		for (int i = 0; i < n; i++)
			v[i] = a*x[i] + b*y[i];
	}

	static void scalar_mult(vector<double> &v, double a, const vector<double> &x)
	{
		int n = x.size();
		v.resize(n);
		for (int i = 0; i < n; i++)
			v[i] = a*x[i];
	}

	static double dot(const vector<double> &x, const vector<double> &y)
	{
		int n = x.size();
		double d = 0;
		for (int i = 0; i < n; i++)
			d += x[i] * y[i];
		return d;
	}
}
