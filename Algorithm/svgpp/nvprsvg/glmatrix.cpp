
/* glmatrix.cpp - OpenGL-style matrix utilities based on cg4cpp types */

// Copyright (c) NVIDIA Corporation. All rights reserved.

// Matrix convention: row major (C style arrays)

#include <Cg/vector/xyzw.hpp>
#include <Cg/matrix/1based.hpp>
#include <Cg/matrix/0based.hpp>
#include <Cg/matrix/rows.hpp>
#include <Cg/matrix/elements.hpp>

#include "glmatrix.hpp"

#include <Cg/normalize.hpp>
#include <Cg/radians.hpp>
#include <Cg/transpose.hpp>
#include <Cg/inverse.hpp>
#include <Cg/mul.hpp>
#include <Cg/max.hpp>
#include <Cg/upper4x4.hpp>

// Grumble, Microsoft (and probably others) define these as macros
#undef min
#undef max

namespace Cg {

template <typename T>
__CGmatrix<T,3,3> rotate3x3(const __CGvector<T,1> & angle, const __CGvector<T,3> & v)
{
    typedef __CGvector<T,3> Vector;
    Vector nv = normalize(v);
    T rad_angle = radians(angle);
    T sine   = sin(rad_angle);
    T cosine = cos(rad_angle);

    Vector abc = nv * nv.yzx * (1-cosine);
    Vector t = nv * nv;
    Vector s = nv * sine;

    Vector v0 = t + cosine*(1-t),
           v1 = abc + s.zxy,
           v2 = abc - s.yzx;

    __CGmatrix<T,3,3> rv;
    rv._m00 = v0.x;
    rv._m11 = v0.y;
    rv._m22 = v0.z;

    rv._m10 = v1.x;
    rv._m21 = v1.y;
    rv._m02 = v1.z;

    rv._m20 = v2.x;
    rv._m01 = v2.y;
    rv._m12 = v2.z;

    return rv;
}

float3x3 rotate3x3(float1 angle, float3 vector)
{
    return rotate3x3<float>(angle, vector);
}

double3x3 rotate3x3(double1 angle, double3 vector)
{
    return rotate3x3<double>(angle, vector);
}

float4x4 rotate4x4(float1 angle, float3 vector)
{
    float3x3 m3 = rotate3x3(angle, vector);
    float4x4 m4 = float4x4(float4(m3[0], 0),
                           float4(m3[1], 0),
                           float4(m3[2], 0),
                           float4(0,0,0, 1));
    return m4;
}

double4x4 rotate4x4(double1 angle, double3 vector)
{
    double3x3 m3 = rotate3x3(angle, vector);
    double4x4 m4 = double4x4(double4(m3[0], 0),
                   double4(m3[1], 0),
                   double4(m3[2], 0),
                   double4(0,0,0, 1));
    return m4;
}

template <typename T>
__CGmatrix<T,2,2> scale2x2(const __CGvector<T,2> & v)
{
    __CGmatrix<T,2,2> rv = 0;
    rv._m00 = v.x;
    rv._m11 = v.y;
    return rv;
}

template <typename T>
__CGmatrix<T,3,3> scale3x3(const __CGvector<T,3> & v)
{
    __CGmatrix<T,3,3> rv = 0;
    rv._m00 = v.x;
    rv._m11 = v.y;
    rv._m22 = v.z;
    return rv;
}

template <typename T>
__CGmatrix<T,3,4> scale3x4(const __CGvector<T,3> & vector)
{
    return upper3x4(scale3x3(vector));
}

template <typename T>
__CGmatrix<T,4,4> scale4x4(const __CGvector<T,2> & vector)
{
    return upper4x4(scale2x2(vector));
}

template <typename T>
__CGmatrix<T,4,4> scale4x4(const __CGvector<T,3> & vector)
{
    return upper4x4(scale3x3(vector));
}

float4x4 scale4x4(float2 v)
{
    return scale4x4<float>(v);
}

float4x4 scale4x4(float3 v)
{
    return scale4x4<float>(v);
}

double4x4 scale4x4(double3 vector)
{
    double3x3 m3 = scale3x3(vector);
    double4x4 m4 = double4x4(double4(m3[0], 0),
                             double4(m3[1], 0),
                             double4(m3[2], 0),
                             double4(0,0,0, 1));
    return m4;
}

template <typename T>
__CGmatrix<T,4,4> translate4x4(const __CGvector<T,3> & v)
{
    __CGmatrix<T,4,4> rv = 0;
    rv._m00 = 1;
    rv._m11 = 1;
    rv._m22 = 1;
    rv._m33 = 1;
    rv._m03 = v.x;
    rv._m13 = v.y;
    rv._m23 = v.z;
    return rv;
}

float1x1 identity1x1()
{
    return float1x1(1);
}

float2x2 identity2x2()
{
    return float2x2(1,0,
                    0,1);
}

float3x3 identity3x3()
{
    return float3x3(1,0,0,
                    0,1,0,
                    0,0,1);
}

float4x4 identity4x4()
{
    return float4x4(1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1);
}

// Math from page 54-56 of "Digital Image Warping" by George Wolberg,
// though credited to Paul Heckert's "Fundamentals of Texture
// Mapping and Image Warping" 1989 Master's thesis.
double3x3 square2quad(const float2 v[4])
{
  double3x3 a;

  double2 d1 = double2(v[1]-v[2]),
          d2 = double2(v[3]-v[2]),
          d3 = double2(v[0]-v[1]+v[2]-v[3]);

  double denom = d1.x*d2.y - d2.x*d1.y;
  // Generate column-major matrix, matching book's math.
  a._13 = (d3.x*d2.y - d2.x*d3.y) / denom;
  a._23 = (d1.x*d3.y - d3.x*d1.y) / denom;
  a._11_12 = v[1] - v[0] + a._13*v[1];
  a._21_22 = v[3] - v[0] + a._23*v[3];
  a._31_32 = v[0];
  a._33 = 1;

  // Return row-major matrix.
  return transpose(a);
}

double3x3 quad2square(const float2 v[4])
{
  return inverse(square2quad(v));
}

double3x3 quad2quad(const float2 from[4], const float2 to[4])
{
  return mul(square2quad(to), quad2square(from));
}

double3x3 box2quad(const float4 &box, const float2 to[4])
{
  float2 from[4] = { box.xy, box.zy, box.zw, box.xw };

  return quad2quad(from, to);
}

float4x4 make_float4x4(const double3x3 &m)
{
    float4x4 rv;

    rv[0] = float4(m[0].xy, 0, m[0].z); 
    rv[1] = float4(m[1].xy, 0, m[1].z);
    rv[2] = float4(0,0,     1, 0);
    rv[3] = float4(m[2].xy, 0, m[2].z);

    return rv;
}

} // namespace Cg

