
/* glmatrix.hpp - prototypes for OpenGL-style cg4cpp matrix utilities */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __glmatrix_hpp__
#define __glmatrix_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <Cg/double.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

namespace Cg {

template <typename T>
__CGmatrix<T,3,3> rotate3x3(const __CGvector<T,1> & angle, const __CGvector<T,3> & v);
float3x3 rotate3x3(float1 angle, float3 vector);
double3x3 rotate3x3(double1 angle, double3 vector);
float4x4 rotate4x4(float1 angle, float3 vector);
double4x4 rotate4x4(double1 angle, double3 vector);
template <typename T>
__CGmatrix<T,3,3> scale3x3(const __CGvector<T,3> & v);
float4x4 scale4x4(float2 vector);
float4x4 scale4x4(float3 vector);
double4x4 scale4x4(double3 vector);
template <typename T>
__CGmatrix<T,4,4> translate4x4(const __CGvector<T,3> & v);

float1x1 identity1x1();
float2x2 identity2x2();
float3x3 identity3x3();
float4x4 identity4x4();

double3x3 square2quad(const float2 v[4]);
double3x3 quad2square(const float2 v[4]);
double3x3 quad2quad(const float2 from[4], const float2 to[4]);
double3x3 box2quad(const float4 &box, const float2 to[4]);

float4x4 make_float4x4(const double3x3 &m);

} // namespace Cg

#endif // __glmatrix_hpp__
