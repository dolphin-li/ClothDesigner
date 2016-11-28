
/* path.hpp - path rendering class. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#ifndef __path_hpp__
#define __path_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <vector>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>

#include <stdio.h>    // for printf and NULL
#include <stdlib.h>   // for exit
#include <math.h>     // for sin and cos
#include "GL/glew.h"
#if __APPLE__
#include <OpenGL/glext.h>
#endif

#include <Cg/double.hpp>
#include <Cg/vector/xyzw.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

#include "path_stats.hpp"
#include "path_process.hpp"

#if defined(_MSC_VER)
# pragma warning(disable: 4355) /* 'this' : used in base member initializer list */
#endif

// Grumble, Microsoft's <windef.h> (and probably other headers) define these as macros
#undef min
#undef max

using namespace Cg;
using namespace boost;
using std::string;

using std::vector;

struct Command {
    Command() { }
    Command(char _cmd, int _ndx) : cmd(_cmd), ndx(_ndx) { }

    char cmd;
    int ndx;
};

typedef shared_ptr<struct Path> PathPtr;
typedef shared_ptr<struct Node> NodePtr;

#include "ActiveControlPoint.hpp"
#include "PathStyle.hpp"

typedef shared_ptr<RendererState<Path> > PathRendererStatePtr;

struct LdpPolyGroup : enable_shared_from_this<LdpPolyGroup> {
public:
	float3 color;
	std::vector<int> cmds;

	LdpPolyGroup(){
		color.x = color.y = color.z = 0;
	}
};

typedef boost::shared_ptr<LdpPolyGroup> LdpPolyGroupPtr;

struct Path : enable_shared_from_this<Path>, HasRendererState<Path> {
private:
    bool has_logical_bbox;
    float4 logical_bbox;

public:
    // Path data
    vector<char> cmd;
    vector<float> coord;

    PathStyle style;
	int ldp_poly_id;
	std::vector<int> ldp_corner_ids;
	float ldp_poly_3dCenter[3];
	float ldp_poly_3dRot[4];
	bool ldp_poly_cylinder_dir;

    Path(const PathStyle &style, const char *string);
    Path(const PathStyle &style, const vector<char> &cmds, const vector<float> &coords);
    Path(const char *string);
    Path(const vector<char> &cmds, const vector<float> &coords);

    void invalidate();

    Path & operator = (const Path &src) {
        if (this != &src) {
            assert(!"rethink");

            this->cmd = src.cmd;
            this->coord = src.coord;
            this->style = src.style;

            this->has_logical_bbox = src.has_logical_bbox;
            this->logical_bbox = src.logical_bbox;

            // Empty the renderer_state array
            this->renderer_states = vector<RendererStatePtr>();
        }
        return *this;
    }

    bool isEmpty();
    bool isFillable() {
        return style.do_fill;
    }
    bool isStrokable() {
        return style.do_stroke;
    }

    string convert_to_svg_path(const float4x4 &transform);

    void drawControlPoints();
    void drawReferencePoints();

    void testForControlPointHit(ActiveControlPoint &hit,
                                float2 &p,
                                size_t coord_index,
                                float2 &relative_to_coord,
                                ActiveControlPoint::CoordUsage coord_usage,
                                bool &hit_path);
    void findNearerControlPoint(ActiveControlPoint &hit);

    int countSegments();

    float4 getActualFillBounds();   // computes bounding box for filled region of path
    float4 getDilatedFillBounds();  // dilates actual fill bounds by stroke width, if stroking
    float4 getBounds();             // substitute logical bounds, if specified, for dilated fill bounds
    void gatherStats(PathStats &stats);

    void setLogicalBounds(const float4 &bbox);
    void unsetLogicalBounds();

    // Iterate over all the path's segments, determining the
    // segment data, and calling the appropriate segment processor
    // virtual function for the segment type (moveto, lineto, etc.).
    void processSegments(PathSegmentProcessor &processor);

private:
    void validate();
    void fillValidate();
    void strokeValidate();
} ;

#endif // __path_hpp__
