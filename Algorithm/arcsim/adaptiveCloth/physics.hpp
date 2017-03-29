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
#include "cloth.hpp"
#include "geometry.hpp"
#include "simulation.hpp"
#include <vector>
namespace arcsim
{
	template <Space s>
	double internal_energy(const Cloth &cloth);

	double constraint_energy(const std::vector<Constraint*> &cons);

	double external_energy(const Cloth &cloth, const Vec3 &gravity,
		const Wind &wind);

	// A += dt^2 dF/dx; b += dt F + dt^2 dF/dx v
	// also adds damping terms
	// if dt == 0, just does A += dF/dx; b += F instead, no damping
	template <Space s>
	void add_internal_forces(const Cloth &cloth, SpMat<Mat3x3> &A,
		std::vector<Vec3> &b, double dt);

	void add_constraint_forces(const Cloth &cloth,
		const std::vector<Constraint*> &cons,
		SpMat<Mat3x3> &A, std::vector<Vec3> &b, double dt);

	void add_external_forces(const Cloth &cloth, const Vec3 &gravity,
		const Wind &wind, std::vector<Vec3> &fext,
		std::vector<Mat3x3> &Jext);

	void add_morph_forces(const Cloth &cloth, const Morph &morph, double t,
		double dt,
		std::vector<Vec3> &fext, std::vector<Mat3x3> &Jext);

	void implicit_update(Cloth &cloth, const std::vector<Vec3> &fext,
		const std::vector<Mat3x3> &Jext,
		const std::vector<Constraint*> &cons, double dt,
		bool update_positions = true);

}
