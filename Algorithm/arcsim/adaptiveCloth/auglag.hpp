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

#pragma once
namespace arcsim
{

#include "sparse.hpp"
#include "vectors.hpp"

	struct NLOpt
	{ // nonlinear optimization problem
		// minimize objective s.t. constraints = or <= 0
		int nvar, ncon;
		virtual void initialize(double *x) const = 0;
		virtual void precompute(const double *x) const {}
		virtual double objective(const double *x) const = 0;
		virtual void obj_grad(const double *x, double *grad) const = 0; // set
		virtual double constraint(const double *x, int j, int &sign) const = 0;
		virtual void con_grad(const double *x, int j, double factor,
			double *grad) const = 0; // add factor*gradient
		virtual void finalize(const double *x) const = 0;
	};

	void augmented_lagrangian_method(const NLOpt &problem, bool verbose = false);

	// convenience functions for when optimization variables are Vec3-valued

	inline Vec3 get_subvec(const double *x, int i)
	{
		return Vec3(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
	}
	inline void set_subvec(double *x, int i, const Vec3 &xi)
	{
		for (int j = 0; j < 3; j++) x[i * 3 + j] = xi[j];
	}
	inline void add_subvec(double *x, int i, const Vec3 &xi)
	{
		for (int j = 0; j < 3; j++) x[i * 3 + j] += xi[j];
	}

}
