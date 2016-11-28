
/* misc_math.h - various inline math routines */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __misc_math_h__
#define __misc_math_h__

#include <Cg/vector.hpp>
#include <Cg/all.hpp>

static inline bool isZero(double v)
{
#if 0
    const double eps = 6e-008;

    if (fabs(v) < eps) {
        return true;
    } else {
        return false;
    }
#else
    return v == 0.0;
#endif
}

static inline bool sameVertex(double3 a, double3 b)
{
    bool3 sameComponents = (a == b);
    bool allSame = all(sameComponents);
    return allSame;
}

static inline bool sameVertex(float2 a, float2 b)
{
    bool2 sameComponents = (a == b);
    bool allSame = all(sameComponents);
    return allSame;
}

static inline bool sameVertex(double2 a, double2 b)
{
    bool2 sameComponents = (a == b);
    bool allSame = all(sameComponents);
    return allSame;
}

static inline double squared_distance(double2 vector)
{
    return dot(vector,vector);
}

#endif // __misc_math_h__
