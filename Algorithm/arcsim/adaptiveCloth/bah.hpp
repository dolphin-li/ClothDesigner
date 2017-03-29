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
	struct Box
	{
		Vec2 umin, umax;
		Box() : umin(Vec2(infinity)), umax(Vec2(-infinity)) {}
		Box(const Vec2 &u) : umin(u), umax(u) {}
		Box &operator+= (const Vec2 &u);
		Box &operator+= (const Box &box);
		bool overlaps(const Box &box) const;
		Vec2 size() const;
		Vec2 center() const;
	};

	struct BahNode
	{
		Box box;
		Face *face;
		BahNode *parent;
		BahNode *left;
		BahNode *right;
		BahNode();
		BahNode(BahNode*, Face*, const Box&);
		BahNode(BahNode*, Face**, unsigned int, Box*, Vec2*);
		~BahNode();
	};

	BahNode *new_bah_tree(const Mesh &mesh);
	void delete_bah_tree(BahNode *root);

	typedef void(*BahCallback) (Face *face0, const Face *face1);
	void for_overlapping_faces(Face *face, const BahNode *node,
		BahCallback callback);

}
