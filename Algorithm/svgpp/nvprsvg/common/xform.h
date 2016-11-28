#ifndef XFORM_H
#define XFORM_H

/* xform.h - C transform routines for path rendering */

#ifdef __cplusplus
extern "C" {
#endif

typedef float Transform3x2[2][3];

extern void identity(Transform3x2 dst);
extern void mul(Transform3x2 dst, Transform3x2 a, Transform3x2 b);
extern void translate(Transform3x2 dst, float x, float y);
extern void scale(Transform3x2 dst, float x, float y);
extern void rotate(Transform3x2 dst, float angle);
extern void ortho(Transform3x2 dst, float l, float r, float b, float t);
extern void inverse_ortho(Transform3x2 dst, float l, float r, float b, float t);

extern void xform(float dst[2], Transform3x2 a, const float v[2]);

extern void MatrixLoadToGL(Transform3x2 m);

#ifdef __cplusplus
}
#endif

#endif /* XFORM_H */
