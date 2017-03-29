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

	// v = x + y
	static void add(vector<double> &v, const vector<double> &x,
		const vector<double> &y);
	// v = a x + b y
	static void add(vector<double> &v, double a, const vector<double> &x,
		double b, const vector<double> &y);
	// v = a x
	static void scalar_mult(vector<double> &v, double a, const vector<double> &x);

	static double dot(const vector<double> &x, const vector<double> &y);
	static double dot(const vector<double> &x, const SpMat<double> &A,
		const vector<double> &y);
	static double norm(const vector<double> &x);

	bool minimize_in_ball(const vector<double> &g, const SpMat<double> &H,
		double radius, vector<double> &p);

	void trust_region_method(const NLOpt &problem, OptOptions opt, bool verbose)
	{
		// see Nocedal and Wright 2006, algorithm 4.1
		int n = problem.nvar;
		double radius = 1e-3;
		double eta = 0.125;
		vector<double> x(n), p(n), x_new(n);
		problem.initialize(&x[0]);
		problem.precompute(&x[0]);
		double f = problem.objective(&x[0]);
		vector<double> g(n);
		SpMat<double> H(n, n);
		assert(problem.hessian(&x[0], H));
		for (int iter = 0; iter < opt.max_iter(); iter++)
		{
			problem.gradient(&x[0], &g[0]);
			if (norm(g) < opt.eps_g())
				break;
			problem.hessian(&x[0], H);
			bool boundary = minimize_in_ball(g, H, radius, p);
			add(x_new, x, p);
			problem.precompute(&x_new[0]);
			double f_new = problem.objective(&x_new[0]);
			double delta_f = f_new - f;
			double delta_m = dot(g, p) + dot(p, H, p) / 2;
			double ratio = delta_f / delta_m;
			radius = (ratio < 0.25) ? radius / 4
				: (ratio > 0.75 && boundary) ? 2 * radius
				: radius;
			if (ratio > eta)
			{
				x.swap(x_new);
				if (radius < opt.eps_x() || f - f_new < opt.eps_f())
					break;
				f = f_new;
				problem.precompute(&x[0]);
			}
		}
		problem.finalize(&x[0]);
	}

	// finds t such that ||x1 + t (x2 - x1)|| = r
	// where n1 = ||x1||, n2 = ||x2||, d = x1 dot x2
	double line_circle_intersection(double n1, double n2, double d, double r);

	bool minimize_in_ball(const vector<double> &g, const SpMat<double> &H,
		double radius, vector<double> &p)
	{
		int n = g.size();
		static vector<double> p1, p2;
		scalar_mult(p1, -dot(g, g) / dot(g, H, g), g);
		p2 = taucs_linear_solve(H, g);
		scalar_mult(p2, -1, p2);
		double n1 = norm(p1), n2 = norm(p2);
		if (n2 < radius)
		{
			p.swap(p2);
			return false;
		}
		else if (n1 < radius)
		{
			double d12 = dot(p1, p2);
			double t = line_circle_intersection(n1, n2, d12, radius);
			add(p, 1 - t, p1, t, p2);
			return true;
		}
		else
		{
			scalar_mult(p, radius / n1, p1);
			return true;
		}
	}

	double line_circle_intersection(double n1, double n2, double d, double r)
	{
		double a = sq(n1) - 2 * d + sq(n2),
			b = -2 * sq(n1) + 2 * d,
			c = sq(n1) - sq(r);
		double x[2];
		int nsol = solve_quadratic(a, b, c, x);
		return (nsol < 2) ? 0 : (x[0] >= 0) ? x[0] : x[1];
	}

	static void add(vector<double> &v, const vector<double> &x,
		const vector<double> &y)
	{
		int n = x.size();
		v.resize(n);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
			v[i] = x[i] + y[i];
	}

	static void add(vector<double> &v, double a, const vector<double> &x,
		double b, const vector<double> &y)
	{
		int n = x.size();
		v.resize(n);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
			v[i] = a*x[i] + b*y[i];
	}

	static void scalar_mult(vector<double> &v, double a, const vector<double> &x)
	{
		int n = x.size();
		v.resize(n);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
			v[i] = a*x[i];
	}

	static double dot(const vector<double> &x, const vector<double> &y)
	{
		int n = x.size();
		double d = 0;
#pragma omp parallel for reduction(+:d)
		for (int i = 0; i < n; i++)
			d += x[i] * y[i];
		return d;
	}

	static double dot(const vector<double> &x, const SpMat<double> &A,
		const vector<double> &y)
	{
		int n = x.size();
		double d = 0;
#pragma omp parallel for reduction(+:d)
		for (int i = 0; i < n; i++)
		{
			double Ayi = 0;
			const SpVec<double> &Ai = A.rows[i];
			for (int jj = 0; jj < Ai.indices.size(); jj++)
			{
				int j = Ai.indices[jj];
				double Aij = Ai.entries[jj];
				Ayi += Aij*y[j];
			}
			d += x[i] * Ayi;
		}
		return d;
	}

	static double norm(const vector<double> &x)
	{
		return sqrt(dot(x, x));
	}
}