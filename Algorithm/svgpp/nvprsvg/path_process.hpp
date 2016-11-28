
/* path_process.hpp - path processing */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __path_process_hpp__
#define __path_process_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <Cg/vector.hpp>

#include <Cg/cos.hpp>
#include <Cg/sin.hpp>

using namespace Cg;

struct EndPointArc {
    float2 p[2],
           radii;
    float x_axis_rotation;
    bool large_arc_flag,
         sweep_flag;
};

struct CenterPointArc {
    enum {
        BEHAVED,          // draw an arc
        DEGENERATE_LINE,  // draw a line connecting p0 and p1
        DEGENERATE_POINT  // don't draw anything (p0 == p1)
    } form;
    float2 center;
    float2 radii;
    float2 p[2];
    float theta1;
    float delta_theta;
    float psi;

    CenterPointArc(const EndPointArc &src);

    CenterPointArc() {};

    // Evaluate a point on the arc at a given theta.
    float2 eval(double theta);
};

struct PathSegmentProcessor {
    virtual void beginPath(PathPtr p) = 0;
    virtual void moveTo(const float2 p[2], size_t coord_index, char cmd) = 0;
    virtual void lineTo(const float2 p[2], size_t coord_index, char cmd) = 0;
    virtual void quadraticCurveTo(const float2 p[3], size_t coord_index, char cmd) = 0;
    virtual void cubicCurveTo(const float2 p[4], size_t coord_index, char cmd) = 0;
    virtual void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) = 0;
    virtual void close(char cmd) = 0;
    virtual void endPath(PathPtr p) = 0;
    virtual ~PathSegmentProcessor() {}
};

#endif // __path_process_hpp__
