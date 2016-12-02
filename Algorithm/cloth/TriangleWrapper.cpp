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
			if (panel.outerPoly())
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
		m_triBuffer.clear();
		m_triVertsBuffer.clear();
	}

	void TriangleWrapper::precomputeSewing()
	{
		m_sampleParams.clear();
		m_shapePieceMap.clear();

		// 1. calc basic nums, without considering the sewings
		const float step = m_triSize;
		for (auto& piece : (*m_pieces))
		{
			const auto& panel = piece->panel();
			if (panel.outerPoly())
			for (const auto& shape : *panel.outerPoly())
			{
				addSampleParam(shape.get(), step / shape->getLength());
				m_shapePieceMap[shape.get()] = piece.get();
			}
			for (const auto& dart : panel.darts())
			for (const auto& shape : *dart)
			{
				addSampleParam(shape.get(), step / shape->getLength());
				m_shapePieceMap[shape.get()] = piece.get();
			}
			for (const auto& line : panel.innerLines())
			for (const auto& shape : *line)
			{
				addSampleParam(shape.get(), step / shape->getLength());
				m_shapePieceMap[shape.get()] = piece.get();
			}
		} // end for piece
	}

	void TriangleWrapper::addSampleParam(const AbstractShape* shape, float step)
	{
		auto iter = m_sampleParams.find(shape);
		if (iter != m_sampleParams.end())
		{
			if (iter->second->step <= step)
				return;
			iter->second->params.clear();
		}
		else
		{
			m_sampleParams.insert(std::make_pair(shape, SampleParamVecPtr(new SampleParamVec())));
			iter = m_sampleParams.find(shape);
		}
		iter->second->step = step;
		for (float s = 0.f; s < 1 + step - 1e-8; s += step)
			iter->second->params.push_back(SampleParam(std::min(1.f, s), 0));
	}

	void TriangleWrapper::addPolygon(const ShapeGroup& poly)
	{
		const float step = m_triSize;
		const float thre = m_ptMergeThre;
		int startIdx = (int)m_points.size();

		// add points
		for (const auto& shape : poly)
		{
			auto& spVec = m_sampleParams[shape.get()];
			for (auto& sp : spVec->params)
			{
				auto p = shape->getPointByParam(sp.t);
				if (m_points.size() != startIdx)
				{
					if ((m_points.back() - p).length() < thre || (m_points[startIdx] - p).length() < thre)
					{
						sp.idx = int(m_points.size()) - 1; // merged to the last point
						continue;
					}
				}
				sp.idx = int(m_points.size());
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
		if (m_points.size() < 3)
			return;
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
		sprintf_s(m_cmds, "Qpzq%da%fYY", 30, triAreaWanted);
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

		mesh2d.clear();
		for (const auto& v : m_triVertsBuffer)
			mesh2d.vertex_list.push_back(ldp::Float3(v[0], v[1], 0));
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

	static int findParamRange(float t, const std::vector<float>& ranges)
	{
		int bg = 0, ed = (int)ranges.size()-1;
		while (bg < ed)
		{
			float tb = ranges[bg];
			float te = ranges[ed];
			if (t <= te && t >= tb && ed - bg == 1)
				return bg;
			int mid = (bg + ed) / 2;
			float tm = ranges[mid];
			if (t <= tm && t >= tb)
				ed = mid;
			else if (t >= tm && t <= te)
				bg = mid;
			else
				return -1;
		}
		return -1;
	}

	void TriangleWrapper::postComputeSewing()
	{
		// 1. remove invalid samples
		std::vector<SampleParam> tmpVec;
		for (auto& iter : m_sampleParams)
		{
			auto& params = iter.second->params;
			tmpVec = params;
			params.clear();
			for (const auto& p : tmpVec)
			{
				if (p.idx >= 0)
					params.push_back(p);
			}
		} // end for iter

		// 2. count the verts of each piece and accumulate
		m_vertStart.clear();
		int vertNum = 0;
		for (const auto& piece : *m_pieces)
		{
			m_vertStart[piece.get()] = vertNum;
			vertNum += (int)piece->mesh2d().vertex_list.size();
		}

		// 2. make stitching
		m_stitches.clear();
		std::vector<StitchPoint> fpts, spts;
		std::vector<float> fLens, sLens, fParams;
		for (const auto& sew : *m_sewings)
		{
			if (sew->empty())
				continue;
			const auto& firsts = sew->firsts();
			const auto& seconds = sew->seconds();

			// update param
			fLens.clear();
			sLens.clear();
			fLens.push_back(0);
			sLens.push_back(0);
			for (const auto& f : firsts)
				fLens.push_back(fLens.back() + ((const AbstractShape*)Sewing::getPtrById(f.id))->getLength());
			for (const auto& s : seconds)
				sLens.push_back(sLens.back() + ((const AbstractShape*)Sewing::getPtrById(s.id))->getLength());
			for (auto& f : fLens)
				f /= fLens.back();
			for (auto& s : sLens)
				s /= sLens.back();

			// sampel points now
			fpts.clear();
			spts.clear();
			fParams.clear();
			for (size_t iUnit = 0; iUnit < firsts.size(); iUnit++)
			{
				const auto& unit = firsts[iUnit];
				const float paramStart = fLens[iUnit];
				const float paramLen = fLens[iUnit + 1] - paramStart;
				auto shape = (AbstractShape*)Sewing::getPtrById(unit.id);
				auto piece = m_shapePieceMap[shape];
				const int vStart = m_vertStart[piece];
				auto params = m_sampleParams[shape]->params;
				if (unit.reverse)
				{
					std::reverse(params.begin(), params.end());
					for (auto& p : params)
						p.t = 1.f - p.t;
				}
				for (const auto& pm : params)
				{
					StitchPoint s;		
					s.vids = Int2(vStart + pm.idx, vStart + pm.idx);
					s.w = 0.f;
					fpts.push_back(s);
					fParams.push_back(pm.t*paramLen + paramStart);
				} // end for pm
			} // end for firsts
			
			for (const auto& tGlobal : fParams)
			{
				int iUnit = findParamRange(tGlobal, sLens);
				assert(iUnit >= 0);
				const auto& unit = seconds[iUnit];
				const float paramStart = sLens[iUnit];
				const float paramLen = sLens[iUnit + 1] - paramStart;
				assert(tGlobal >= paramStart && tGlobal <= paramStart + paramLen);
				float tLocal = (tGlobal - paramStart) / paramLen;
				auto shape = (AbstractShape*)Sewing::getPtrById(unit.id);
				auto piece = m_shapePieceMap[shape];
				const int vStart = m_vertStart[piece];
				auto params = m_sampleParams[shape]->params;
				if (unit.reverse)
				{
					std::reverse(params.begin(), params.end());
					for (auto& p : params)
						p.t = 1.f - p.t;
				}
				assert(params.size() > 1);
				bool found = false;
				for (size_t i = 1; i < params.size(); i++)
				{
					const auto& ps = params[i - 1];
					const auto& pe = params[i];
					if (!(pe.t >= tLocal && ps.t <= tLocal))
						continue;
					StitchPoint s;
					s.vids = Int2(vStart + ps.idx, vStart + pe.idx);
					s.w = (tLocal - ps.t) / (pe.t - ps.t);
					spts.push_back(s);
					found = true;
					break;
				} // end for pm
				if (!found)
				{
					printf("warning: a stitch may not be made proper!\n");
					StitchPoint s;
					s.w = -1;
					spts.push_back(s);
				}
			} // end for t

			assert(fpts.size() == spts.size());
			for (size_t i = 0; i < fpts.size(); i++)
			{
				if (spts[i].w < 0)
					continue;
				m_stitches.push_back(StitchPointPair(fpts[i], spts[i]));
			}
		} // end for sew
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

