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

#include "geometry.hpp"
#include <cstdlib>

using namespace std;

namespace arcsim
{

	double signed_vf_distance(const Vec3 &x,
		const Vec3 &y0, const Vec3 &y1, const Vec3 &y2,
		Vec3 *n, double *w)
	{
		Vec3 _n; if (!n) n = &_n;
		double _w[4]; if (!w) w = _w;
		*n = cross(normalize(y1 - y0), normalize(y2 - y0));
		if (norm2(*n) < 1e-6)
			return infinity;
		*n = normalize(*n);
		double h = dot(x - y0, *n);
		double b0 = stp(y1 - x, y2 - x, *n),
			b1 = stp(y2 - x, y0 - x, *n),
			b2 = stp(y0 - x, y1 - x, *n);
		w[0] = 1;
		w[1] = -b0 / (b0 + b1 + b2);
		w[2] = -b1 / (b0 + b1 + b2);
		w[3] = -b2 / (b0 + b1 + b2);
		return h;
	}

	double signed_ee_distance(const Vec3 &x0, const Vec3 &x1,
		const Vec3 &y0, const Vec3 &y1,
		Vec3 *n, double *w)
	{
		Vec3 _n; if (!n) n = &_n;
		double _w[4]; if (!w) w = _w;
		*n = cross(normalize(x1 - x0), normalize(y1 - y0));
		if (norm2(*n) < 1e-6)
			return infinity;
		*n = normalize(*n);
		double h = dot(x0 - y0, *n);
		double a0 = stp(y1 - x1, y0 - x1, *n), a1 = stp(y0 - x0, y1 - x0, *n),
			b0 = stp(x0 - y1, x1 - y1, *n), b1 = stp(x1 - y0, x0 - y0, *n);
		w[0] = a0 / (a0 + a1);
		w[1] = a1 / (a0 + a1);
		w[2] = -b0 / (b0 + b1);
		w[3] = -b1 / (b0 + b1);
		return h;
	}

	bool set_unsigned_ve_distance(const Vec3 &x, const Vec3 &y0, const Vec3 &y1,
		double *_d, Vec3 *_n,
		double *_wx, double *_wy0, double *_wy1)
	{
		double t = clamp(dot(x - y0, y1 - y0) / dot(y1 - y0, y1 - y0), 0., 1.);
		Vec3 y = y0 + t*(y1 - y0);
		double d = norm(x - y);
		if (d < *_d)
		{
			*_d = d;
			*_n = normalize(x - y);
			*_wx = 1;
			*_wy0 = 1 - t;
			*_wy1 = t;
			return true;
		}
		return false;
	}

	bool set_unsigned_vf_distance(const Vec3 &x,
		const Vec3 &y0, const Vec3 &y1, const Vec3 &y2,
		double *_d, Vec3 *_n,
		double *_wx,
		double *_wy0, double *_wy1, double *_wy2)
	{
		Vec3 n = normalize(cross(normalize(y1 - y0), normalize(y2 - y0)));
		double d = abs(dot(x - y0, n));
		double b0 = stp(y1 - x, y2 - x, n),
			b1 = stp(y2 - x, y0 - x, n),
			b2 = stp(y0 - x, y1 - x, n);
		if (d < *_d && b0 >= 0 && b1 >= 0 && b2 >= 0)
		{
			*_d = d;
			*_n = n;
			*_wx = 1;
			*_wy0 = -b0 / (b0 + b1 + b2);
			*_wy1 = -b1 / (b0 + b1 + b2);
			*_wy2 = -b2 / (b0 + b1 + b2);
			return true;
		}
		bool success = false;
		if (b0 < 0
			&& set_unsigned_ve_distance(x, y1, y2, _d, _n, _wx, _wy1, _wy2))
		{
			success = true;
			*_wy0 = 0;
		}
		if (b1 < 0
			&& set_unsigned_ve_distance(x, y2, y0, _d, _n, _wx, _wy2, _wy0))
		{
			success = true;
			*_wy1 = 0;
		}
		if (b2 < 0
			&& set_unsigned_ve_distance(x, y0, y1, _d, _n, _wx, _wy0, _wy1))
		{
			success = true;
			*_wy2 = 0;
		}
		return success;
	}

	bool set_unsigned_ee_distance(const Vec3 &x0, const Vec3 &x1,
		const Vec3 &y0, const Vec3 &y1,
		double *_d, Vec3 *_n,
		double *_wx0, double *_wx1,
		double *_wy0, double *_wy1)
	{
		Vec3 n = normalize(cross(normalize(x1 - x0), normalize(y1 - y0)));
		double d = abs(dot(x0 - y0, n));
		double a0 = stp(y1 - x1, y0 - x1, n), a1 = stp(y0 - x0, y1 - x0, n),
			b0 = stp(x0 - y1, x1 - y1, n), b1 = stp(x1 - y0, x0 - y0, n);
		if (d < *_d && a0 >= 0 && a1 >= 0 && b0 >= 0 && b1 >= 0)
		{
			*_d = d;
			*_n = n;
			*_wx0 = a0 / (a0 + a1);
			*_wx1 = a1 / (a0 + a1);
			*_wy0 = -b0 / (b0 + b1);
			*_wy1 = -b1 / (b0 + b1);
			return true;
		}
		bool success = false;
		if (a0 < 0
			&& set_unsigned_ve_distance(x1, y0, y1, _d, _n, _wx1, _wy0, _wy1))
		{
			success = true;
			*_wx0 = 0;
		}
		if (a1 < 0
			&& set_unsigned_ve_distance(x0, y0, y1, _d, _n, _wx0, _wy0, _wy1))
		{
			success = true;
			*_wx1 = 0;
		}
		if (b0 < 0
			&& set_unsigned_ve_distance(y1, x0, x1, _d, _n, _wy1, _wx0, _wx1))
		{
			success = true;
			*_wy0 = 0;
			*_n = -*_n;
		}
		if (b1 < 0
			&& set_unsigned_ve_distance(y0, x0, x1, _d, _n, _wy0, _wx0, _wx1))
		{
			success = true;
			*_wy1 = 0;
			*_n = -*_n;
		}
		return success;
	}

	double unsigned_vf_distance(const Vec3 &x,
		const Vec3 &y0, const Vec3 &y1, const Vec3 &y2,
		Vec3 *n, double w[4])
	{
		Vec3 _n; if (!n) n = &_n;
		double _w[4]; if (!w) w = _w;
		double d = infinity;
		set_unsigned_vf_distance(x, y0, y1, y2, &d, n, &w[0], &w[1], &w[2], &w[3]);
		return d;
	}

	double unsigned_ee_distance(const Vec3 &x0, const Vec3 &x1,
		const Vec3 &y0, const Vec3 &y1,
		Vec3 *n, double w[4])
	{
		Vec3 _n; if (!n) n = &_n;
		double _w[4]; if (!w) w = _w;
		double d = infinity;
		set_unsigned_ee_distance(x0, x1, y0, y1, &d, n, &w[0], &w[1], &w[2], &w[3]);
		return d;
	}

	Vec3 get_barycentric_coords(const Vec2& point, const Face* f)
	{
		// Compute vectors        
		Vec2 v0 = f->v[0]->u - f->v[2]->u;
		Vec2 v1 = f->v[1]->u - f->v[2]->u;
		Vec2 v2 = point - f->v[2]->u;
		// Compute dot products
		double dot00 = dot(v0, v0);
		double dot01 = dot(v0, v1);
		double dot02 = dot(v0, v2);
		double dot11 = dot(v1, v1);
		double dot12 = dot(v1, v2);
		// Compute barycentric coordinates
		double invDenom = 1.f / (dot00 * dot11 - dot01 * dot01);
		double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		return Vec3(u, v, 1 - u - v);
	}

	// Is the point within the face?
	// Adapted from http://www.blackpawn.com/texts/pointinpoly/default.html
	bool is_inside(const Vec2& point, const Face* f)
	{
		Vec3 bary = get_barycentric_coords(point, f);
		//printf("UV: %f, %f\n", u, v);
		// Check if point is in triangle
		// 10*epsilon: want to be robust for borders
		return ((bary[0] >= -10 * EPSILON) && (bary[1] >= -10 * EPSILON) && (bary[2] >= -100 * EPSILON));
	}

	// Gets the face that surrounds point u in material space
	Face* get_enclosing_face(const Mesh& mesh, const Vec2& u,
		Face *starting_face_hint)
	{
		for (int f = 0; f < mesh.faces.size(); f++)
		if (is_inside(u, mesh.faces[f]))
			return mesh.faces[f];
		return NULL;
	}

	template <> const Vec3 &pos<PS>(const Node *node) { return node->y; }
	template <> const Vec3 &pos<WS>(const Node *node) { return node->x; }
	template <> Vec3 &pos<PS>(Node *node) { return node->y; }
	template <> Vec3 &pos<WS>(Node *node) { return node->x; }

	template <Space s> Vec3 nor(const Face *face)
	{
		const Vec3 &x0 = pos<s>(face->v[0]->node),
			&x1 = pos<s>(face->v[1]->node),
			&x2 = pos<s>(face->v[2]->node);
		return normalize(cross(x1 - x0, x2 - x0));
	}
	template Vec3 nor<PS>(const Face *face);
	template Vec3 nor<WS>(const Face *face);

	double unwrap_angle(double theta, double theta_ref)
	{
		if (theta - theta_ref > M_PI)
			theta -= 2 * M_PI;
		if (theta - theta_ref < -M_PI)
			theta += 2 * M_PI;
		return theta;
	}

	template <Space s> double dihedral_angle(const Edge *edge)
	{
		// if (!hinge.edge[0] || !hinge.edge[1]) return 0;
		// const Edge *edge0 = hinge.edge[0], *edge1 = hinge.edge[1];
		// int s0 = hinge.s[0], s1 = hinge.s[1];
		if (!edge->adjf[0] || !edge->adjf[1])
			return 0;
		Vec3 e = normalize(pos<s>(edge->n[0]) - pos<s>(edge->n[1]));
		if (norm2(e) == 0) return 0;
		Vec3 n0 = nor<s>(edge->adjf[0]), n1 = nor<s>(edge->adjf[1]);
		if (norm2(n0) == 0 || norm2(n1) == 0) return 0;
		double cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
		double theta = atan2(sine, cosine);
		return unwrap_angle(theta, edge->reference_angle);
	}
	template double dihedral_angle<PS>(const Edge *edge);
	template double dihedral_angle<WS>(const Edge *edge);

	template <Space s> Mat2x2 curvature(const Face *face)
	{
		Mat2x2 S;
		for (int e = 0; e < 3; e++)
		{
			const Edge *edge = face->adje[e];
			Vec2 e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
				t_mat = perp(normalize(e_mat));
			double theta = dihedral_angle<s>(face->adje[e]);
			S -= 1 / 2.*theta*norm(e_mat)*outer(t_mat, t_mat);
		}
		S /= face->a;
		return S;
	}
	template Mat2x2 curvature<PS>(const Face *face);
	template Mat2x2 curvature<WS>(const Face *face);
}