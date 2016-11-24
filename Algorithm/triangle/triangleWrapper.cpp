#include "triangleWrapper.h"

extern "C"{
#include "triangle.h"
};

TriangleWrapper::TriangleWrapper()
{}

TriangleWrapper::~TriangleWrapper()
{}

static void destroyMem_trianglulateio(struct triangulateio *io)
{
	if (io->pointlist)  free(io->pointlist);                                               /* In / out */
	if (io->pointattributelist) free(io->pointattributelist);                                      /* In / out */
	if (io->pointmarkerlist) free(io->pointmarkerlist);                                          /* In / out */

	if (io->trianglelist) free(io->trianglelist);                                             /* In / out */
	if (io->triangleattributelist) free(io->triangleattributelist);                                   /* In / out */
	if (io->trianglearealist) free(io->trianglearealist);                                    /* In only */
	if (io->neighborlist) free(io->neighborlist);                                           /* Out only */

	if (io->segmentlist) free(io->segmentlist);                                              /* In / out */
	if (io->segmentmarkerlist) free(io->segmentmarkerlist);                             /* In / out */

	if (io->holelist) free(io->holelist);                        /* In / pointer to array copied out */

	if (io->regionlist) free(io->regionlist);                      /* In / pointer to array copied out */

	if (io->edgelist) free(io->edgelist);                                                 /* Out only */
	if (io->edgemarkerlist) free(io->edgemarkerlist);           /* Not used with Voronoi diagram; out only */
	if (io->normlist) free(io->normlist);              /* Used only with Voronoi diagram; out only */

	//all set to 0
	init_trianglulateio(io);
}

static void init_triangulateIO_from_poly(triangulateio *pIO, const std::vector<ldp::Double2> &poly, int close_flag)
{
	init_trianglulateio(pIO);
	pIO->numberofpoints = (int)poly.size();
	pIO->pointlist = (REAL *)malloc(pIO->numberofpoints * 2 * sizeof(REAL));
	for (int i = 0; i<pIO->numberofpoints; i++)
	{
		pIO->pointlist[i * 2] = poly[i][0];
		pIO->pointlist[i * 2 + 1] = poly[i][1];
	}
	/*********input segments************/
	pIO->numberofsegments = (int)poly.size() - 1;
	if (close_flag) pIO->numberofsegments++;
	pIO->segmentlist = (int *)malloc(pIO->numberofsegments * 2 * sizeof(int));
	for (int i = 0; i<(int)poly.size() - 1; i++)
	{
		pIO->segmentlist[i * 2] = i;
		pIO->segmentlist[i * 2 + 1] = i + 1;
	}
	if (close_flag)
	{
		pIO->segmentlist[pIO->numberofsegments * 2 - 2] = (int)poly.size() - 1;
		pIO->segmentlist[pIO->numberofsegments * 2 - 1] = 0;
	}
}

static int tess_poly_2D(const std::vector<ldp::Double2> &poly,
	std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3> &vTriangle,
	int close_flag, double triAreaWanted)
{
	vTriangle.resize(0);

	triangulateio input, triout, vorout;

	// Define input points
	init_triangulateIO_from_poly(&input, poly, close_flag);

	// Make necessary initializations so that Triangle can return a 
	// triangulation in `mid' and a voronoi diagram in `vorout'.  
	init_trianglulateio(&triout);
	init_trianglulateio(&vorout);

	// triangulate
	char cmd[1024];
	// p: polygon mode
	// z: zero based indexing
	// q%d: minimum angle %d
	// D: delauney
	// a%f: maximum triangle area %f
	sprintf_s(cmd, "pzq%dDa%f", 30, triAreaWanted);
	triangulate(cmd, &input, &triout, &vorout);
	vTriangle.resize(triout.numberoftriangles);
	memcpy(vTriangle.data(), triout.trianglelist, sizeof(int)*triout.numberoftriangles * 3);
	const ldp::Double2* vptr = (const ldp::Double2*)triout.pointlist;
	triVerts.resize(triout.numberofpoints);
	for (int i = 0; i < triout.numberofpoints; i++)
		triVerts[i] = vptr[i];
	destroyMem_trianglulateio(&input);
	destroyMem_trianglulateio(&triout);
	destroyMem_trianglulateio(&vorout);
	return vTriangle.size();
}

void TriangleWrapper::triangulate(const std::vector<ldp::Float2>& polyVerts,
	std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3>& tris,
	int close_flag, float triAreaWanted)
{
	std::vector<ldp::Double2> polyVertsD;
	polyVertsD.assign(polyVerts.begin(), polyVerts.end());
	tess_poly_2D(polyVertsD, triVerts, tris, close_flag, triAreaWanted);
}
