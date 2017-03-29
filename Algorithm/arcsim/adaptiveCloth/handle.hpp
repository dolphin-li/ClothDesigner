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
#include "constraint.hpp"
#include "mesh.hpp"
#include <vector>
namespace arcsim
{
	struct Handle
	{
		double start_time, end_time, fade_time;
		virtual ~Handle() {};
		virtual std::vector<Constraint*> get_constraints(double t) = 0;
		virtual std::vector<Node*> get_nodes() = 0;
		bool active(double t) { return t >= start_time && t <= end_time; }
		double strength(double t)
		{
			if (t < start_time || t > end_time + fade_time) return 0;
			if (t <= end_time) return 1;
			double s = 1 - (t - end_time) / (fade_time + 1e-6);
			return sq(sq(s));
		}
	};

	struct NodeHandle : public Handle
	{
		Node *node;
		const Motion *motion;
		bool activated;
		Vec3 x0;
		NodeHandle() : activated(false) {}
		std::vector<Constraint*> get_constraints(double t);
		std::vector<Node*> get_nodes() { return std::vector<Node*>(1, node); }
	};

	struct CircleHandle : public Handle
	{
		Mesh *mesh;
		int label;
		const Motion *motion;
		double c; // circumference
		Vec2 u;
		Vec3 xc, dx0, dx1;
		std::vector<Constraint*> get_constraints(double t);
		std::vector<Node*> get_nodes() { return std::vector<Node*>(); }
	};

	struct GlueHandle : public Handle
	{
		Node* nodes[2];
		std::vector<Constraint*> get_constraints(double t);
		std::vector<Node*> get_nodes()
		{
			std::vector<Node*> ns;
			ns.push_back(nodes[0]);
			ns.push_back(nodes[1]);
			return ns;
		}
	};

}
