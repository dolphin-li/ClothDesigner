#ifndef XFORM_HPP
#define XFORM_HPP

/* xform.hpp - C++ transform routines for path rendering */

#include <Cg/double.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

extern Cg::float3x3 ortho(float l, float r, float b, float t);
extern Cg::float3x3 inverse_ortho(float l, float r, float b, float t);
extern Cg::float3x3 translate(float x, float y);
extern Cg::float3x3 scale(float x, float y);
extern Cg::float3x3 rotate(float angle);

extern Cg::double3x3 square2quad(const Cg::float2 v[4]);
extern Cg::double3x3 quad2square(const Cg::float2 v[4]);
extern Cg::double3x3 quad2quad(const Cg::float2 from[4], const Cg::float2 to[4]);
extern Cg::double3x3 box2quad(const Cg::float4 &box, const Cg::float2 to[4]);

extern void MatrixMultToGL(Cg::float3x3 m);
extern void MatrixLoadToGL(Cg::float3x3 m);

#endif /* XFORM_HPP */
