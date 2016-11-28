
/* xform.c - C transform routines for path rendering */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <math.h>

#include <GL/glew.h>

#include "xform.h"

void identity(Transform3x2 dst)
{
  dst[0][0] = 1;
  dst[0][1] = 0;
  dst[0][2] = 0;
  dst[1][0] = 0;
  dst[1][1] = 1;
  dst[1][2] = 0;
}

void mul(Transform3x2 dst, Transform3x2 a, Transform3x2 b)
{
  Transform3x2 result;

  result[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0];
  result[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
  result[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2];

  result[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0];
  result[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
  result[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2];

  dst[0][0] = result[0][0];
  dst[0][1] = result[0][1];
  dst[0][2] = result[0][2];
  dst[1][0] = result[1][0];
  dst[1][1] = result[1][1];
  dst[1][2] = result[1][2];
}

void translate(Transform3x2 dst, float x, float y)
{
  dst[0][0] = 1;
  dst[0][1] = 0;
  dst[0][2] = x;
  dst[1][0] = 0;
  dst[1][1] = 1;
  dst[1][2] = y;
}

void scale(Transform3x2 dst, float x, float y)
{
  dst[0][0] = x;
  dst[0][1] = 0;
  dst[0][2] = 0;

  dst[1][0] = 0;
  dst[1][1] = y;
  dst[1][2] = 0;
}

void rotate(Transform3x2 dst, float angle)
{
  float radians = angle*3.14159f/180.0f,
        s = (float)sin(radians),
        c = (float)cos(radians);

  dst[0][0] = c;
  dst[0][1] = -s;
  dst[0][2] = 0;
  dst[1][0] = s;
  dst[1][1] = c;
  dst[1][2] = 0;
}

void ortho(Transform3x2 dst, float l, float r, float b, float t)
{
  dst[0][0] = 2/(r-l);
  dst[0][1] = 0;
  dst[0][2] = -(r+l)/(r-l);
  dst[1][0] = 0;
  dst[1][1] = 2/(t-b);
  dst[1][2] = -(t+b)/(t-b);
}

void inverse_ortho(Transform3x2 dst, float l, float r, float b, float t)
{
  dst[0][0] = (r-l)/2;
  dst[0][1] = 0;
  dst[0][2] = (r+l)/2;
  dst[1][0] = 0;
  dst[1][1] = (t-b)/2;
  dst[1][2] = (t+b)/2;
}

void xform(float dst[2], Transform3x2 a, const float v[2])
{
  float result[2];

  result[0] = a[0][0]*v[0] + a[0][1]*v[1] + a[0][2];
  result[1] = a[1][0]*v[0] + a[1][1]*v[1] + a[1][2];

  dst[0] = result[0];
  dst[1] = result[1];
}

void MatrixLoadToGL(Transform3x2 m)
{
  GLfloat mm[16];  /* Column-major OpenGL-style 4x4 matrix. */

  /* First column. */
  mm[0] = m[0][0];
  mm[1] = m[1][0];
  mm[2] = 0;
  mm[3] = 0;

  /* Second column. */
  mm[4] = m[0][1];
  mm[5] = m[1][1];
  mm[6] = 0;
  mm[7] = 0;

  /* Third column. */
  mm[8] = 0;
  mm[9] = 0;
  mm[10] = 1;
  mm[11] = 0;

  /* Fourth column. */
  mm[12] = m[0][2];
  mm[13] = m[1][2];
  mm[14] = 0;
  mm[15] = 1;

  glMatrixLoadfEXT(GL_MODELVIEW, &mm[0]);
}
