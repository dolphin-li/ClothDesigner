#include "TriangleWrapper.h"
#include "ldpMat\ldp_basic_mat.h"
#include "clothPiece.h"
#include "panelPolygon.h"
#include "TransformInfo.h"
#include "Renderable\ObjMesh.h"
extern "C"{
#include "triangle\triangle.h"
};

namespace ldp
{
	TriangleWrapper::TriangleWrapper()
	{
		m_in = new triangulateio;
		m_out = new triangulateio;
		m_vro = new triangulateio;
		init_trianglulateio(m_in);
		init_trianglulateio(m_out);
		init_trianglulateio(m_vro);
		m_pieces = nullptr;
		m_sewings = nullptr;
		m_ptMergeThre = 0;
		m_ptOnLineThre = 0;
		m_triSize = 0;
	}

	TriangleWrapper::~TriangleWrapper()
	{
		reset_triangle_struct(m_in);
		reset_triangle_struct(m_out);
		reset_triangle_struct(m_vro);
		delete m_in;
		delete m_out;
		delete m_vro;
	}

	void TriangleWrapper::triangulate(
		std::vector<std::shared_ptr<ClothPiece>>& pieces,
		std::vector<std::shared_ptr<Sewing>>& sewings,
		float pointMergeThre,
		float triangleSize,
		float pointOnLineThre
		)
	{
		m_pieces = &pieces;
		m_sewings = &sewings;
		m_ptMergeThre = pointMergeThre;
		m_ptOnLineThre = pointOnLineThre;
		m_triSize = triangleSize;

		precomputeSewing();
		for (auto& piece : (*m_pieces))
		{
			const auto& panel = piece->panel();
			prepareTriangulation();
			addPolygon(*panel.outerPoly().get());
			for (const auto& dart : panel.darts())
				addDart(*dart.get());
			for (const auto& line : panel.innerLines())
				addLine(*line.get());
			finalizeTriangulation();
			generateMesh(*piece.get());
		} // end for piece
		postComputeSewing();
	}

	void TriangleWrapper::prepareTriangulation()
	{
		reset_triangle_struct(m_in);
		reset_triangle_struct(m_out);
		reset_triangle_struct(m_vro);
		m_points.clear();
		m_segments.clear();
		m_holeCenters.clear();
	}

	void TriangleWrapper::precomputeSewing()
	{

	}

	void TriangleWrapper::addPolygon(const ShapeGroup& poly)
	{
		const float step = m_triSize;
		const float thre = m_ptMergeThre;
		int startIdx = (int)m_points.size();

		// add points
		for (const auto& shape : poly)
		{
			const auto& pts = shape->samplePointsOnShape(step / shape->getLength());
			for (const auto& p : pts)
			{
				if (m_points.size() != startIdx)
				{
					if ((m_points.back() - p).length() < thre || (m_points[startIdx] - p).length() < thre)
						continue;
				}
				m_points.push_back(p);
			} // end for p
		} // end for shape

		// add segments
		for (int i = startIdx; i < (int)m_points.size()-1; i++)
			m_segments.push_back(Int2(i, i+1));
		if ((int)m_points.size() > startIdx)
			m_segments.push_back(Int2((int)m_points.size()-1, startIdx));
	}

	void TriangleWrapper::addDart(const ShapeGroup& dart)
	{

	}

	void TriangleWrapper::addLine(const ShapeGroup& line)
	{

	}

	void TriangleWrapper::finalizeTriangulation()
	{
		// init points
		m_in->numberofpoints = (int)m_points.size();
		if (m_in->numberofpoints)
			m_in->pointlist = (REAL *)malloc(m_in->numberofpoints * 2 * sizeof(REAL));
		for (int i = 0; i<m_in->numberofpoints; i++)
		{
			m_in->pointlist[i * 2] = m_points[i][0];
			m_in->pointlist[i * 2 + 1] = m_points[i][1];
		}
		
		// init segments
		m_in->numberofsegments = (int)m_segments.size();
		if (m_in->numberofsegments)
			m_in->segmentlist = (int *)malloc(m_in->numberofsegments * 2 * sizeof(int));
		for (int i = 0; i<(int)m_segments.size(); i++)
		{
			m_in->segmentlist[i * 2] = m_segments[i][0];
			m_in->segmentlist[i * 2 + 1] = m_segments[i][1];
		}

		// init holes
		m_in->numberofholes = (int)m_holeCenters.size();
		if (m_in->numberofholes)
			m_in->holelist = (REAL*)malloc(m_in->numberofholes * 2 * sizeof(REAL));
		for (int i = 0; i<m_in->numberofholes; i++)
		{
			m_in->holelist[i * 2] = m_holeCenters[i][0];
			m_in->holelist[i * 2 + 1] = m_holeCenters[i][1];
		}

		// perform triangulation
		const float triAreaWanted = 0.5 * m_triSize * m_triSize;
		// Q: quiet
		// p: polygon mode
		// z: zero based indexing
		// q%d: minimum angle %d
		// D: delauney
		// a%f: maximum triangle area %f
		// YY: do not allow additional points inserted on segment
		sprintf_s(m_cmds, "Qpzq%dDa%fYY", 30, triAreaWanted);
		::triangulate(m_cmds, m_in, m_out, m_vro);
		m_triBuffer.resize(m_out->numberoftriangles);
		memcpy(m_triBuffer.data(), m_out->trianglelist, sizeof(int)*m_out->numberoftriangles * 3);
		const ldp::Double2* vptr = (const ldp::Double2*)m_out->pointlist;
		m_triVertsBuffer.resize(m_out->numberofpoints);
		for (int i = 0; i < m_out->numberofpoints; i++)
			m_triVertsBuffer[i] = vptr[i];
	}

	void TriangleWrapper::generateMesh(ClothPiece& piece)
	{
		auto& mesh2d = piece.mesh2d();
		auto& mesh3d = piece.mesh3d();
		auto& mesh3dInit = piece.mesh3dInit();
		auto& transInfo = piece.transformInfo();

		mesh2d.vertex_list.clear();
		for (const auto& v : m_triVertsBuffer)
			mesh2d.vertex_list.push_back(ldp::Float3(v[0], v[1], 0));
		mesh2d.face_list.clear();
		mesh2d.material_list.clear();
		mesh2d.material_list.push_back(ObjMesh::obj_material());
		for (const auto& t : m_triBuffer)
		{
			ObjMesh::obj_face f;
			f.vertex_count = 3;
			f.material_index = 0;
			for (int k = 0; k < f.vertex_count; k++)
				f.vertex_index[k] = t[k];
			mesh2d.face_list.push_back(f);
		}
		mesh2d.updateNormals();
		mesh2d.updateBoundingBox();
		mesh3dInit.cloneFrom(&mesh2d);
		transInfo.apply(mesh3dInit);
		mesh3d.cloneFrom(&mesh3dInit);
	}

	void TriangleWrapper::postComputeSewing()
	{

	}

	void TriangleWrapper::reset_triangle_struct(triangulateio* io)const
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
}

