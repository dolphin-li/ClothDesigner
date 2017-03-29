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
#include "mesh.hpp"
#include "util.hpp"
namespace arcsim
{



	double signed_vf_distance(const Vec3 &x,
		const Vec3 &y0, const Vec3 &y1, const Vec3 &y2,
		Vec3 *n, double *w);

	double signed_ee_distance(const Vec3 &x0, const Vec3 &x1,
		const Vec3 &y0, const Vec3 &y1,
		Vec3 *n, double *w);

	double unsigned_vf_distance(const Vec3 &x,
		const Vec3 &y0, const Vec3 &y1, const Vec3 &y2,
		Vec3 *n, double *w);

	double unsigned_ee_distance(const Vec3 &x0, const Vec3 &x1,
		const Vec3 &y0, const Vec3 &y1,
		Vec3 *n, double *w);

	Vec3 get_barycentric_coords(const Vec2 &point, const Face *face);

	Face* get_enclosing_face(const Mesh& mesh, const Vec2& u,
		Face *starting_face_hint = NULL);

	enum Space { PS, WS }; // plastic space, world space

	template <Space s> const Vec3 &pos(const Node *node);
	template <Space s> Vec3 &pos(Node *node);
	template <Space s> Vec3 nor(const Face *face);
	template <Space s> double dihedral_angle(const Edge *edge);
	template <Space s> Mat2x2 curvature(const Face *face);

	double unwrap_angle(double theta, double theta_ref);

}
