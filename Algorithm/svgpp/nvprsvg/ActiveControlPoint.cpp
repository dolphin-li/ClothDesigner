
// ActiveControlPoint.cpp - selecting an active path control point

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "path.hpp"
#include "scene.hpp"
#include "ActiveControlPoint.hpp"

#include <Cg/mul.hpp>
#include <Cg/inverse.hpp>

#ifdef NDEBUG
const static int verbose = 0;
#else
const static int verbose = 1;
#endif

ActivePoint::ActivePoint()
{
}

ActivePoint::ActivePoint(float2 xy, float4x4 trans, int update_mask_, float closeness_)
{
    mouse_xy = xy;
    current_transform = trans;
    closeness = closeness_;
    update_mask = update_mask_;
    needs_inverse = true;
}

//// ActiveControlPoint

ActiveControlPoint::ActiveControlPoint()
{
}

ActiveControlPoint::ActiveControlPoint(float2 xy, float4x4 trans, int update_mask_, float closeness_)
    : ActivePoint(xy, trans, update_mask_, closeness_)
{
    path = PathPtr();
    coord_index = ~0;
    relative_to_coord = float2(0,0);
    coord_usage = NONE;
}

ActiveControlPoint & ActiveControlPoint::operator = (const ActiveControlPoint & src)
{
    if (this != &src) {
        mouse_xy = src.mouse_xy;
        closeness = src.closeness;
        current_transform = src.current_transform;
        update_mask = src.update_mask;
        path = src.path;
        coord_index = src.coord_index;
        relative_to_coord = src.relative_to_coord;
        coord_usage = src.coord_usage;
        hit_transform = src.hit_transform;
        needs_inverse = src.needs_inverse;
        hit_inv_transform = src.hit_inv_transform;
    }
    return *this;
}

void ActiveControlPoint::set(float2 window_space_xy)
{
    float4 p = float4(window_space_xy, 0, 1);

    if (needs_inverse) {
        hit_inv_transform = inverse(hit_transform);
        needs_inverse = false;
    }

    p = mul(hit_inv_transform, p);
    float2 path_xy = p.xy / p.w;

    switch (coord_usage) {
    case X_AND_Y:
        path->coord[coord_index+0] = path_xy.x - relative_to_coord.x;
        path->coord[coord_index+1] = path_xy.y - relative_to_coord.y;
        break;
    case X_ONLY:
        path->coord[coord_index+0] = path_xy.x - relative_to_coord.x;
        break;
    default:
        assert(!"bogus coord_usage");
    case Y_ONLY:
        path->coord[coord_index+0] = path_xy.y - relative_to_coord.y;
        break;
    }
    path->invalidate();
}

// Given a point, test if it is closer than closest hit, and if so, update ActiveControlPoint.
void ActiveControlPoint::testForControlPointHit(
                                const float2 &p,
                                size_t new_coord_index,
                                const float2 &new_relative_to_coord,
                                ActiveControlPoint::CoordUsage new_coord_usage,
                                bool &hit_path,
                                PathPtr &new_path)
{
    float4 xyzw_window = mul(current_transform, float4(p,0,1));
    float2 xy_window = xyzw_window.xy / xyzw_window.w;
    float d = distance(mouse_xy, xy_window);
    if (d < closeness) {
        if (!hit_path) {
            path = new_path;
            hit_transform = current_transform;
            hit_path = true;
        }
        closeness = d;
        coord_index = new_coord_index;
        relative_to_coord = new_relative_to_coord;
        coord_usage = new_coord_usage;
    }
}

//// ActiveWarpPoint

ActiveWarpPoint::ActiveWarpPoint()
{
}

ActiveWarpPoint::ActiveWarpPoint(float2 xy, float4x4 trans, int update_mask_, float closeness_)
    : ActivePoint(xy, trans, update_mask_, closeness_)
{
    transform = WarpTransformPtr();
    ndx = ~0;  // bogus
}

ActiveWarpPoint & ActiveWarpPoint::operator = (const ActiveWarpPoint & src)
{
    if (this != &src) {
        mouse_xy = src.mouse_xy;
        closeness = src.closeness;
        current_transform = src.current_transform;
        update_mask = src.update_mask;
        transform = src.transform;
        ndx = src.ndx;
        hit_transform = src.hit_transform;
        needs_inverse = src.needs_inverse;
        hit_inv_transform = src.hit_inv_transform;
    }
    return *this;
}

void ActiveWarpPoint::set(float2 window_space_xy)
{
    float4 p = float4(window_space_xy, 0, 1);

    if (needs_inverse) {
        hit_inv_transform = inverse(hit_transform);
        needs_inverse = false;
    }

    p = mul(hit_inv_transform, p);
    float2 transform_xy = p.xy / p.w;

    transform->setWarpPoint(ndx, transform_xy);
}

// Given a point, test if it is closer than closest hit, and if so, update ActiveControlPoint.
void ActiveWarpPoint::testForControlPointHit(
                                const float2 &p,
                                int new_ndx,
                                bool &got_transform,
                                WarpTransformPtr new_transform)
{
    float4 xyzw_window = mul(current_transform, float4(p,0,1));
    float2 xy_window = xyzw_window.xy / xyzw_window.w;
    float d = distance(mouse_xy, xy_window);
    if (verbose) {
        std::cout << "top = " << current_transform << std::endl;
        std::cout << "p = " << p << std::endl;
        std::cout << "d = " << d << std::endl;
        std::cout << "xy_window = " << xy_window << std::endl;
        std::cout << "mouse_xy = " << mouse_xy << std::endl;
    }
    if (d < closeness) {
        if (!got_transform) {
            transform = new_transform;
            hit_transform = current_transform;
            got_transform = true;
        }
        closeness = d;
        ndx = new_ndx;
    }
}
