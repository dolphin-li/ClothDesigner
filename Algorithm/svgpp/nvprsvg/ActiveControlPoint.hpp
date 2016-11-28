
/* ActiveControlPoint.hpp - active control point. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __ActiveControlPoint_hpp__
#define __ActiveControlPoint_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/shared_ptr.hpp>

#include <Cg/double.hpp>
#include <Cg/vector/xyzw.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

using namespace boost;
using namespace Cg;

typedef shared_ptr<struct Path> PathPtr;
typedef shared_ptr<struct WarpTransform> WarpTransformPtr;

struct ActivePoint {
    float2 mouse_xy;
    float closeness;
    int update_mask;

    float4x4 current_transform;

    float4x4 hit_transform;
    bool needs_inverse;
    float4x4 hit_inv_transform;

    ActivePoint();
    ActivePoint(float2 xy, float4x4 trans, int update_mask, float closeness = 10);
};

struct ActiveControlPoint : ActivePoint {
    PathPtr path;
    enum CoordUsage {
        NONE,
        X_AND_Y,
        X_ONLY,
        Y_ONLY,
    };
    CoordUsage coord_usage;
    float2 relative_to_coord;
    size_t coord_index;

    ActiveControlPoint();
    ActiveControlPoint(float2 xy, float4x4 trans, int update_mask, float closeness = 10);
    ActiveControlPoint & operator = (const ActiveControlPoint & src);
    void set(float2 window_space_xy);

    void testForControlPointHit(const float2 &p,
                                size_t new_coord_index,
                                const float2 &new_relative_to_coord,
                                ActiveControlPoint::CoordUsage new_coord_usage,
                                bool &hit_path,
                                PathPtr &new_path);
};

struct ActiveWarpPoint : ActivePoint {
    WarpTransformPtr transform;
    int ndx;

    ActiveWarpPoint();
    ActiveWarpPoint(float2 xy, float4x4 trans, int update_mask, float closeness = 10);
    ActiveWarpPoint & operator = (const ActiveWarpPoint & src);
    void set(float2 window_space_xy);

    void testForControlPointHit(const float2 &p,
                                int new_ndx,
                                bool &hit_transform,
                                WarpTransformPtr new_transform);
};

#endif // __ActiveControlPoint_hpp__
