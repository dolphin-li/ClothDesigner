/************************************************************************/
/* NOTE: THIS file is only for bmesh.cpp, you cannot include it directly
/************************************************************************/
#if defined(_MSC_VER)
#  define __func__ __FUNCTION__
#endif
#define BG_DISK_EDGE_LINK_GET(e, v)  (                                        \
	((v) == ((BGEdge *)(e))->getStartPoint()) ?                                            \
	&((e)->v1_disk_link) :                                                \
	&((e)->v2_disk_link)                                                  \
	)
#define MAKE_ID(a,b,c,d) ( (int)(d)<<24 | (int)(c)<<16 | (b)<<8 | (a) )
#define FREEWORD MAKE_ID('f', 'r', 'e', 'e')
#define MEMPOOL_ELEM_SIZE_MIN (sizeof(void *) * 2)
#define MIN2(x,y)               ( (x)<(y) ? (x) : (y) )
#define MIN3(x,y,z)             MIN2( MIN2((x),(y)) , (z) )
#define MIN4(x,y,z,a)           MIN2( MIN2((x),(y)) , MIN2((z),(a)) )

#define MAX2(x,y)               ( (x)>(y) ? (x) : (y) )
#define MAX3(x,y,z)             MAX2( MAX2((x),(y)) , (z) )
#define MAX4(x,y,z,a)           MAX2( MAX2((x),(y)) , MAX2((z),(a)) )
#define INIT_MINMAX(min, max) {                                               \
	(min)[0]= (min)[1]= (min)[2]= 1.0e30f;                                \
	(max)[0]= (max)[1]= (max)[2]= -1.0e30f;                               \
}
#define INIT_MINMAX2(min, max) {                                              \
	(min)[0]= (min)[1]= 1.0e30f;                                          \
	(max)[0]= (max)[1]= -1.0e30f;                                         \
} (void)0
#define DO_MIN(vec, min) {                                                    \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (min)[2]>(vec)[2] ) (min)[2]= (vec)[2];                           \
} (void)0
#define DO_MAX(vec, max) {                                                    \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
	if( (max)[2]<(vec)[2] ) (max)[2]= (vec)[2];                           \
} (void)0
#define DO_MINMAX(vec, min, max) {                                            \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (min)[2]>(vec)[2] ) (min)[2]= (vec)[2];                           \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
	if( (max)[2]<(vec)[2] ) (max)[2]= (vec)[2];                           \
} (void)0
#define DO_MINMAX2(vec, min, max) {                                           \
	if( (min)[0]>(vec)[0] ) (min)[0]= (vec)[0];                           \
	if( (min)[1]>(vec)[1] ) (min)[1]= (vec)[1];                           \
	if( (max)[0]<(vec)[0] ) (max)[0]= (vec)[0];                           \
	if( (max)[1]<(vec)[1] ) (max)[1]= (vec)[1];                           \
} (void)0
#ifndef SWAP
#define SWAP(type, a, b)	{ type sw_ap; sw_ap=(a); (a)=(b); (b)=sw_ap; }
#endif
#define BG_FACE_FIRST_LOOP(p) ((p)->l_first)
#define UNLIKELY(e) (e)
#define LIKELY(e) (e)
#define BGESH_ASSERT(e) assert(e)
#undef NULL
#undef TRUE
#undef FALSE
#define TRUE 1
#define FALSE 0
#define NULL 0
#ifndef M_PI
#define M_PI        3.14159265358979323846f
#endif
#define SMALL_NUMBER 1e-8f
#define MINLINE static __forceinline
#define BG_NGON_STACK_SIZE 32
#define BG_LOOPS_OF_FACE 11
#ifndef M_SQRT2
#define M_SQRT2     1.41421356237309504880
#endif
#define ISECT_LINE_LINE_COLINEAR	-1
#define ISECT_LINE_LINE_NONE		 0
#define ISECT_LINE_LINE_EXACT		 1
#define ISECT_LINE_LINE_CROSS		 2
#define SET_INT_IN_POINTER(i)    ((void *)(intptr_t)(i))
#define GET_INT_FROM_POINTER(i)  ((int)(intptr_t)(i))

MINLINE void mul_v3_fl(float r[3], float f)
{
	r[0] *= f;
	r[1] *= f;
	r[2] *= f;
}
MINLINE void mul_v2_fl(float r[3], float f)
{
	r[0] *= f;
	r[1] *= f;
}

MINLINE void mul_v3_v3fl(float r[3], const float a[3], float f)
{
	r[0] = a[0] * f;
	r[1] = a[1] * f;
	r[2] = a[2] * f;
}
MINLINE void mul_v3_v3(float r[3], const float a[3])
{
	r[0] *= a[0];
	r[1] *= a[1];
	r[2] *= a[2];
}
MINLINE void sub_v3_v3(float r[3], const float a[3])
{
	r[0] -= a[0];
	r[1] -= a[1];
	r[2] -= a[2];
}
MINLINE void madd_v3_v3fl(float r[3], const float a[3], float f)
{
	r[0] += a[0] * f;
	r[1] += a[1] * f;
	r[2] += a[2] * f;
}

MINLINE void madd_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] += a[0] * b[0];
	r[1] += a[1] * b[1];
	r[2] += a[2] * b[2];
}

MINLINE void sub_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[0] - b[0];
	r[1] = a[1] - b[1];
	r[2] = a[2] - b[2];
}
MINLINE void add_v3_v3(float r[3], const float a[3])
{
	r[0] += a[0];
	r[1] += a[1];
	r[2] += a[2];
}
MINLINE void madd_v3_v3v3fl(float r[3], const float a[3], const float b[3], float f)
{
	r[0] = a[0] + b[0] * f;
	r[1] = a[1] + b[1] * f;
	r[2] = a[2] + b[2] * f;
}

MINLINE void madd_v3_v3v3v3(float r[3], const float a[3], const float b[3], const float c[3])
{
	r[0] = a[0] + b[0] * c[0];
	r[1] = a[1] + b[1] * c[1];
	r[2] = a[2] + b[2] * c[2];
}

MINLINE void add_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[0] + b[0];
	r[1] = a[1] + b[1];
	r[2] = a[2] + b[2];
}
MINLINE void mid_v3_v3v3(float v[3], const float v1[3], const float v2[3])
{
	v[0] = 0.5f * (v1[0] + v2[0]);
	v[1] = 0.5f * (v1[1] + v2[1]);
	v[2] = 0.5f * (v1[2] + v2[2]);
}
MINLINE void negate_v3(float r[3])
{
	r[0] = -r[0];
	r[1] = -r[1];
	r[2] = -r[2];
}

MINLINE void negate_v3_v3(float r[3], const float a[3])
{
	r[0] = -a[0];
	r[1] = -a[1];
	r[2] = -a[2];
}

MINLINE int compare_v3v3(const float v1[3], const float v2[3], const float limit)
{
	if (fabsf(v1[0] - v2[0]) < limit)
		if (fabsf(v1[1] - v2[1]) < limit)
			if (fabsf(v1[2] - v2[2]) < limit)
				return 1;

	return 0;
}
MINLINE void copy_v3_v3(float r[3], const float a[3])
{
	r[0] = a[0];
	r[1] = a[1];
	r[2] = a[2];
}
MINLINE float saasin(float fac)
{
	if (fac <= -1.0f) return (float)-M_PI / 2.0f;
	else if (fac >= 1.0f) return (float)M_PI / 2.0f;
	else return (float)asin(fac);
}
MINLINE float saacos(float fac)
{
	if (fac <= -1.0f) return (float)M_PI;
	else if (fac >= 1.0f) return 0.0;
	else return (float)acos(fac);
}
MINLINE void zero_v3(float r[3])
{
	r[0] = 0.0f;
	r[1] = 0.0f;
	r[2] = 0.0f;
}

MINLINE void cross_v3_v3v3(float r[3], const float a[3], const float b[3])
{
	r[0] = a[1] * b[2] - a[2] * b[1];
	r[1] = a[2] * b[0] - a[0] * b[2];
	r[2] = a[0] * b[1] - a[1] * b[0];
}
MINLINE float cross_v2v2(const float a[2], const float b[2])
{
	return a[0] * b[1] - a[1] * b[0];
}
MINLINE float dot_v3v3(const float* x, const float* y)
{
	return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}
MINLINE float len_v3(const float* d)
{
	return sqrtf(dot_v3v3(d,d));
}
MINLINE float len_v3v3(const float* a, const float*b)
{
	float c[3];
	sub_v3_v3v3(c,a,b);
	return len_v3(c);
}
MINLINE float normalize_v3_v3(float r[3], const float a[3])
{
	float d = dot_v3v3(a, a);

	/* a larger value causes normalize errors in a
	 * scaled down models with camera xtreme close */
	if (d > 1.0e-35f) {
		d = sqrtf(d);
		mul_v3_v3fl(r, a, 1.0f / d);
	}
	else {
		zero_v3(r);
		d = 0.0f;
	}

	return d;
}
MINLINE double normalize_v3_d(double n[3])
{
	double d = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

	/* a larger value causes normalize errors in a
	 * scaled down models with camera xtreme close */
	if (d > 1.0e-35) {
		double mul;

		d = sqrt(d);
		mul = 1.0 / d;

		n[0] *= mul;
		n[1] *= mul;
		n[2] *= mul;
	}
	else {
		n[0] = n[1] = n[2] = 0;
		d = 0.0;
	}

	return d;
}
MINLINE float normalize_v3(float n[3])
{
	return normalize_v3_v3(n, n);
}
MINLINE float angle_normalized_v3v3(const float v1[3], const float v2[3])
{
	/* this is the same as acos(dot_v3v3(v1, v2)), but more accurate */
	if (dot_v3v3(v1, v2) < 0.0f) {
		float vec[3];

		vec[0] = -v2[0];
		vec[1] = -v2[1];
		vec[2] = -v2[2];

		return (float)M_PI - 2.0f * (float)saasin(len_v3v3(vec, v1) / 2.0f);
	}
	else
		return 2.0f * (float)saasin(len_v3v3(v2, v1) / 2.0f);
}
MINLINE float angle_v3v3v3(const float v1[3], const float v2[3], const float v3[3])
{
	float vec1[3], vec2[3];

	sub_v3_v3v3(vec1, v2, v1);
	sub_v3_v3v3(vec2, v2, v3);
	normalize_v3(vec1);
	normalize_v3(vec2);

	return angle_normalized_v3v3(vec1, vec2);
}
MINLINE void add_newell_cross_v3_v3v3(float n[3], const float v_prev[3], const float v_curr[3])
{
	n[0] += (v_prev[1] - v_curr[1]) * (v_prev[2] + v_curr[2]);
	n[1] += (v_prev[2] - v_curr[2]) * (v_prev[0] + v_curr[0]);
	n[2] += (v_prev[0] - v_curr[0]) * (v_prev[1] + v_curr[1]);
}
MINLINE float shell_angle_to_dist(const float angle)
{
	return (angle < SMALL_NUMBER) ? 1.0f : fabsf(1.0f / cosf(angle));
}
MINLINE void unit_qt(float q[4])
{
	q[0] = 1.0f;
	q[1] = q[2] = q[3] = 0.0f;
}

static void axis_angle_to_quat(float q[4], const float axis[3], float angle)
{
	float nor[3];
	float si;

	if (normalize_v3_v3(nor, axis) == 0.0f) {
		unit_qt(q);
		return;
	}

	angle /= 2;
	si = (float)sin(angle);
	q[0] = (float)cos(angle);
	q[1] = nor[0] * si;
	q[2] = nor[1] * si;
	q[3] = nor[2] * si;
}

MINLINE float line_point_side_v2(const float l1[2], const float l2[2], const float pt[2])
{
	return (((l1[0] - pt[0]) * (l2[1] - pt[1])) -
		((l2[0] - pt[0]) * (l1[1] - pt[1])));
}

MINLINE float len_squared_v3v3(const float a[3], const float b[3])
{
	float d[3];

	sub_v3_v3v3(d, b, a);
	return dot_v3v3(d, d);
}
MINLINE void mul_v3_m3v3(float r[3], float M[3][3], float a[3])
{
	r[0] = M[0][0] * a[0] + M[1][0] * a[1] + M[2][0] * a[2];
	r[1] = M[0][1] * a[0] + M[1][1] * a[1] + M[2][1] * a[2];
	r[2] = M[0][2] * a[0] + M[1][2] * a[1] + M[2][2] * a[2];
}

MINLINE void mul_m3_v3(float M[3][3], float r[3])
{
	float tmp[3];

	mul_v3_m3v3(tmp, M, r);
	copy_v3_v3(r, tmp);
}

static void quat_to_mat3(float m[][3], const float q[4])
{
	double q0, q1, q2, q3, qda, qdb, qdc, qaa, qab, qac, qbb, qbc, qcc;

	q0 = M_SQRT2 * (double)q[0];
	q1 = M_SQRT2 * (double)q[1];
	q2 = M_SQRT2 * (double)q[2];
	q3 = M_SQRT2 * (double)q[3];

	qda = q0 * q1;
	qdb = q0 * q2;
	qdc = q0 * q3;
	qaa = q1 * q1;
	qab = q1 * q2;
	qac = q1 * q3;
	qbb = q2 * q2;
	qbc = q2 * q3;
	qcc = q3 * q3;

	m[0][0] = (float)(1.0 - qbb - qcc);
	m[0][1] = (float)(qdc + qab);
	m[0][2] = (float)(-qdb + qac);

	m[1][0] = (float)(-qdc + qab);
	m[1][1] = (float)(1.0 - qaa - qcc);
	m[1][2] = (float)(qda + qbc);

	m[2][0] = (float)(qdb + qac);
	m[2][1] = (float)(-qda + qbc);
	m[2][2] = (float)(1.0 - qaa - qbb);
}

static void cent_tri_v3(float cent[3], const float v1[3], const float v2[3], const float v3[3])
{
	cent[0] = 0.33333f * (v1[0] + v2[0] + v3[0]);
	cent[1] = 0.33333f * (v1[1] + v2[1] + v3[1]);
	cent[2] = 0.33333f * (v1[2] + v2[2] + v3[2]);
}

static void cent_quad_v3(float cent[3], const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	cent[0] = 0.25f * (v1[0] + v2[0] + v3[0] + v4[0]);
	cent[1] = 0.25f * (v1[1] + v2[1] + v3[1] + v4[1]);
	cent[2] = 0.25f * (v1[2] + v2[2] + v3[2] + v4[2]);
}

static float normal_tri_v3(float n[3], const float v1[3], const float v2[3], const float v3[3])
{
	float n1[3], n2[3];

	n1[0] = v1[0] - v2[0];
	n2[0] = v2[0] - v3[0];
	n1[1] = v1[1] - v2[1];
	n2[1] = v2[1] - v3[1];
	n1[2] = v1[2] - v2[2];
	n2[2] = v2[2] - v3[2];
	n[0] = n1[1] * n2[2] - n1[2] * n2[1];
	n[1] = n1[2] * n2[0] - n1[0] * n2[2];
	n[2] = n1[0] * n2[1] - n1[1] * n2[0];

	return normalize_v3(n);
}

static float normal_quad_v3(float n[3], const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	/* real cross! */
	float n1[3], n2[3];

	n1[0] = v1[0] - v3[0];
	n1[1] = v1[1] - v3[1];
	n1[2] = v1[2] - v3[2];

	n2[0] = v2[0] - v4[0];
	n2[1] = v2[1] - v4[1];
	n2[2] = v2[2] - v4[2];

	n[0] = n1[1] * n2[2] - n1[2] * n2[1];
	n[1] = n1[2] * n2[0] - n1[0] * n2[2];
	n[2] = n1[0] * n2[1] - n1[1] * n2[0];

	return normalize_v3(n);
}

static float area_tri_v2(const float v1[2], const float v2[2], const float v3[2])
{
	return 0.5f * fabsf((v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0]));
}

static float area_tri_signed_v2(const float v1[2], const float v2[2], const float v3[2])
{
	return 0.5f * ((v1[0] - v2[0]) * (v2[1] - v3[1]) + (v1[1] - v2[1]) * (v3[0] - v2[0]));
}

/* only convex Quadrilaterals */
static float area_quad_v3(const float v1[3], const float v2[3], const float v3[3], const float v4[3])
{
	float len, vec1[3], vec2[3], n[3];

	sub_v3_v3v3(vec1, v2, v1);
	sub_v3_v3v3(vec2, v4, v1);
	cross_v3_v3v3(n, vec1, vec2);
	len = normalize_v3(n);

	sub_v3_v3v3(vec1, v4, v3);
	sub_v3_v3v3(vec2, v2, v3);
	cross_v3_v3v3(n, vec1, vec2);
	len += normalize_v3(n);

	return (len / 2.0f);
}

/* Triangles */
static float area_tri_v3(const float v1[3], const float v2[3], const float v3[3])
{
	float len, vec1[3], vec2[3], n[3];

	sub_v3_v3v3(vec1, v3, v2);
	sub_v3_v3v3(vec2, v1, v2);
	cross_v3_v3v3(n, vec1, vec2);
	len = normalize_v3(n);

	return (len / 2.0f);
}

static float area_poly_v3(int nr, float verts[][3], const float normal[3])
{
	float x, y, z, area, max;
	float *cur, *prev;
	int a, px = 0, py = 1;

	/* first: find dominant axis: 0==X, 1==Y, 2==Z
	 * don't use 'axis_dominant_v3()' because we need max axis too */
	x = fabsf(normal[0]);
	y = fabsf(normal[1]);
	z = fabsf(normal[2]);
	max = MAX3(x, y, z);
	if (max == y) py = 2;
	else if (max == x) {
		px = 1;
		py = 2;
	}

	/* The Trapezium Area Rule */
	prev = verts[nr - 1];
	cur = verts[0];
	area = 0;
	for (a = 0; a < nr; a++) {
		area += (cur[px] - prev[px]) * (cur[py] + prev[py]);
		prev = verts[a];
		cur = verts[a + 1];
	}

	return fabsf(0.5f * area / max);
}

/* intersect Line-Line, shorts */
static int isect_line_line_v2_int(const int v1[2], const int v2[2], const int v3[2], const int v4[2])
{
	float div, labda, mu;

	div = (float)((v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0]));
	if (div == 0.0f) return ISECT_LINE_LINE_COLINEAR;

	labda = ((float)(v1[1] - v3[1]) * (v4[0] - v3[0]) - (v1[0] - v3[0]) * (v4[1] - v3[1])) / div;

	mu = ((float)(v1[1] - v3[1]) * (v2[0] - v1[0]) - (v1[0] - v3[0]) * (v2[1] - v1[1])) / div;

	if (labda >= 0.0f && labda <= 1.0f && mu >= 0.0f && mu <= 1.0f) {
		if (labda == 0.0f || labda == 1.0f || mu == 0.0f || mu == 1.0f) return ISECT_LINE_LINE_EXACT;
		return ISECT_LINE_LINE_CROSS;
	}
	return ISECT_LINE_LINE_NONE;
}

/* intersect Line-Line, floats */
static int isect_line_line_v2(const float v1[2], const float v2[2], const float v3[2], const float v4[2])
{
	float div, labda, mu;

	div = (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0]);
	if (div == 0.0f) return ISECT_LINE_LINE_COLINEAR;

	labda = ((float)(v1[1] - v3[1]) * (v4[0] - v3[0]) - (v1[0] - v3[0]) * (v4[1] - v3[1])) / div;

	mu = ((float)(v1[1] - v3[1]) * (v2[0] - v1[0]) - (v1[0] - v3[0]) * (v2[1] - v1[1])) / div;

	if (labda >= 0.0f && labda <= 1.0f && mu >= 0.0f && mu <= 1.0f) {
		if (labda == 0.0f || labda == 1.0f || mu == 0.0f || mu == 1.0f) return ISECT_LINE_LINE_EXACT;
		return ISECT_LINE_LINE_CROSS;
	}
	return ISECT_LINE_LINE_NONE;
}

/* point in tri */

static int isect_point_tri_v2(const float pt[2], const float v1[2], const float v2[2], const float v3[2])
{
	if (line_point_side_v2(v1, v2, pt) >= 0.0f) {
		if (line_point_side_v2(v2, v3, pt) >= 0.0f) {
			if (line_point_side_v2(v3, v1, pt) >= 0.0f) {
				return 1;
			}
		}
	}
	else {
		if (!(line_point_side_v2(v2, v3, pt) >= 0.0f)) {
			if (!(line_point_side_v2(v3, v1, pt) >= 0.0f)) {
				return -1;
			}
		}
	}

	return 0;
}

/* point in quad - only convex quads */
static int isect_point_quad_v2(const float pt[2], const float v1[2], const float v2[2], const float v3[2], const float v4[2])
{
	if (line_point_side_v2(v1, v2, pt) >= 0.0f) {
		if (line_point_side_v2(v2, v3, pt) >= 0.0f) {
			if (line_point_side_v2(v3, v4, pt) >= 0.0f) {
				if (line_point_side_v2(v4, v1, pt) >= 0.0f) {
					return 1;
				}
			}
		}
	}
	else {
		if (!(line_point_side_v2(v2, v3, pt) >= 0.0f)) {
			if (!(line_point_side_v2(v3, v4, pt) >= 0.0f)) {
				if (!(line_point_side_v2(v4, v1, pt) >= 0.0f)) {
					return -1;
				}
			}
		}
	}

	return 0;
}


/**
 * \brief COMPUTE POLY NORMAL
 *
 * Computes the normal of a planar
 * polygon See Graphics Gems for
 * computing newell normal.
 */
static void calc_poly_normal(float normal[3], float* verts, int nverts)
{
	float const *v_prev = &verts[3*(nverts - 1)];
	float const *v_curr = &verts[0];
	float n[3] = {0.0f};
	int i;

	/* Newell's Method */
	for (i = 0; i < nverts; v_prev = v_curr, v_curr = &verts[3*(++i)]) {
		add_newell_cross_v3_v3v3(n, v_prev, v_curr);
	}

	if (UNLIKELY(normalize_v3_v3(normal, n) == 0.0f)) {
		normal[2] = 1.0f; /* other axis set to 0.0 */
	}
}
/**
 * COMPUTE POLY PLANE
 *
 * Projects a set polygon's vertices to
 * a plane defined by the average
 * of its edges cross products
 */
static void calc_poly_plane(float(*verts)[3], const int nverts)
{
	
	float avgc[3], norm[3], mag, avgn[3];
	float *v1, *v2, *v3;
	int i;
	
	if (nverts < 3)
		return;

	zero_v3(avgn);
	zero_v3(avgc);

	for (i = 0; i < nverts; i++) {
		v1 = verts[i];
		v2 = verts[(i + 1) % nverts];
		v3 = verts[(i + 2) % nverts];
		normal_tri_v3(norm, v1, v2, v3);

		add_v3_v3(avgn, norm);
	}

	if (UNLIKELY(normalize_v3(avgn) == 0.0f)) {
		avgn[2] = 1.0f;
	}
	
	for (i = 0; i < nverts; i++) {
		v1 = verts[i];
		mag = dot_v3v3(v1, avgn);
		madd_v3_v3fl(v1, avgn, -mag);
	}
}

/**
 * \brief BG LEGAL EDGES
 *
 * takes in a face and a list of edges, and sets to NULL any edge in
 * the list that bridges a concave region of the face or intersects
 * any of the faces's edges.
 */
static void shrink_edgef(float v1[3], float v2[3], const float fac)
{
	float mid[3];

	mid_v3_v3v3(mid, v1, v2);

	sub_v3_v3v3(v1, v1, mid);
	sub_v3_v3v3(v2, v2, mid);

	mul_v3_fl(v1, fac);
	mul_v3_fl(v2, fac);

	add_v3_v3v3(v1, v1, mid);
	add_v3_v3v3(v2, v2, mid);
}


/**
 * \brief POLY ROTATE PLANE
 *
 * Rotates a polygon so that it's
 * normal is pointing towards the mesh Z axis
 */
static void poly_rotate_plane(const float normal[3], float(*verts), const int nverts)
{

	float up[3] = {0.0f, 0.0f, 1.0f}, axis[3], q[4];
	float mat[3][3];
	double angle;
	int i;

	cross_v3_v3v3(axis, normal, up);

	angle = saacos(dot_v3v3(normal, up));

	if (angle == 0.0) return;

	axis_angle_to_quat(q, axis, (float)angle);
	quat_to_mat3(mat, q);

	for (i = 0; i < nverts; i++)
		mul_m3_v3(mat, &verts[i*3]);
}

/************************************************************************/
/* Template Priority Queue
/************************************************************************/
/**
* A minimal heap.
* Two things must be implemented by T:
* operator <, for heap setup.
* function getIndex(), get the original position of elements, for heap position update.
* The index of T must be 0~size.
* */

template<class T>
class LHeap
{
private:
	vector<T*> _heap;		
	vector<int> _pos;
	int size;
	int capacity;
public:
	LHeap()
	{
		_heap.clear();
		_pos.clear();
		size = 0;
		capacity = 0;
	}
	~LHeap()
	{
		release();
	}

	LHeap& init(T* src, int nSrc)
	{
		release();
		capacity = nSrc;
		size = nSrc;
		_heap.resize(size);
		_pos.resize(size);
		for(int i=0; i< size; i++)
		{
			_heap[i]= &src[i];
			_pos[_heap[i]->getIndex()] = i;
		}
		//make heap
		for(int i=((size>>1)-1); i>=0; i--)
		{
			heapDown(i);
		}
		return *this;
	}

	void release()
	{
		size = 0;
		capacity = 0;
		_heap.clear();
		_pos.clear();
	}

	//push elements with insertion sort.
	void push(T* t)			
	{
		assert(size < capacity);
		_heap[size]= t;
		_pos[_heap[size]->getIndex()]=size;
		heapUp(size);
		size++;
	}

	//get the top of the heap
	T* top()
	{
		assert(size > 0);
		return _heap[0];
	}

	//pop the top of heap
	T* pop()
	{
		assert(size > 0);
		T * minElement = _heap[0];
		--size;
		move_heap(size,0);
		if(size>0)
			heapDown(0);
		return minElement;
	}

	bool isEmpty()const
	{
		return size <= 0;
	}

	int getSize()const
	{
		return size;
	}

	bool isInHeap(T* edge)const
	{
		if(!edge)
			return false;
		return _pos[edge->getIndex()]<size && _pos[edge->getIndex()]>=0;
	}

	//I don't check whether "edge" is contained here, so make sure the parameter "edge" is valide.
	T* remove(T* edge)
	{
		assert(size > 0);
		assert(edge->getIndex() >= 0 && edge->getIndex() < capacity);
		int p = _pos[edge->getIndex()];
		T * minElement = _heap[p];
		--size;
		move_heap(size,p);
		if(size>p)
			heapDown(p);
		return minElement;
	}

	//I don't check whether "edge" is contained here, so make sure the parameter "edge" is valide.
	void updatePos(T* edge)
	{
		assert(edge->getIndex() >= 0 && edge->getIndex() < capacity);
		int p = _pos[edge->getIndex()];
		heapUp(p);
		heapDown(p);
	}

private:
	int father(const int& i)const
	{
		return (i-1) >> 1;
	}
	int child1(const int& i)const
	{
		return (i<<1) + 1;
	}
	int child2(const int& i)const
	{
		return (i<<1) + 2;
	}
	void move_heap(int startId, int endId)
	{
		_heap[endId] = _heap[startId];
		_pos[_heap[startId]->getIndex()] = endId;
	}
	void move_heap(T* s, int endId)
	{
		_heap[endId] = s;
		_pos[s->getIndex()] = endId;
	}
	void heapUp(const int& i)
	{
		assert(i>=0 && i<size);
		int k = i;
		T  *ths = _heap[i];
		while(k!=0 && *ths < *_heap[father(k)])
		{
			move_heap(father(k),k);
			k = father(k);
		}
		move_heap(ths, k);
	}
	void heapDown(const int& i)
	{
		assert(i>=0 && i<size);
		int child = i;
		int k = i;
		T  *ths = _heap[i];
		for(k = i; child1(k) < size; k = child)
		{
			//find the smaller child.
			child = child1(k);
			if(child < size-1 && *_heap[child+1] < *_heap[child])
				child++;
			//down one level.
			if(!(*ths < *_heap[child]))
				move_heap(child,k);
			else
				break;
		}
		move_heap(ths, k);
	}
};