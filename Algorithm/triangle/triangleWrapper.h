#pragma once

#include "ldpMat\ldp_basic_mat.h"
class TriangleWrapper
{
public:
	TriangleWrapper();
	~TriangleWrapper();

	// Delauney triangulation of a polygon:
	// input:
	//	@polyVerts: p0, p1, ..., pn
	//	@close_flag: if true, then pn will be connected to p0, the polygon is closed implicitly
	//	@triAreaWanted: the maximum area of each triangle
	// output:
	//	@triVerts:
	static void triangulate(const std::vector<ldp::Float2>& polyVerts, 
		std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3>& tris,
		int close_flag = true, float triAreaWanted = FLT_MAX);

private:
};