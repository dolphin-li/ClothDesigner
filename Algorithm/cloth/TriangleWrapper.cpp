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
		m_sewingVertPairs.clear();

		// 1. calc basic nums, without considering the sewings
		const float step = m_triSize;
		const float thre = m_ptMergeThre;
		for (auto& piece : (*m_pieces))
		{
			const auto& panel = piece->panel();
			if (panel.outerPoly())
			for (const auto& shape : *panel.outerPoly())
				addSampleParam(shape.get(), step / shape->getLength());
			for (const auto& dart : panel.darts())
			for (const auto& shape : *dart)
				addSampleParam(shape.get(), step / shape->getLength());
			for (const auto& line : panel.innerLines())
			for (const auto& shape : *line)
				addSampleParam(shape.get(), step / shape->getLength());
		} // end for piece

		//////////////////////////////////////////////////////////////////////
		/// until this step, we can perform triangulation, but not good sewings...
		//////////////////////////////////////////////////////////////////////
		//return;

		// 2. considering the sewings, adjust the nums
		std::vector<int> sew_valid(m_sewings->size(), 0);
		for (size_t isew = 0; isew < m_sewings->size(); isew++)
		{
			const auto& sew = (*m_sewings)[isew];
			if (sew->empty())
				continue;
			const auto& firsts = sew->firsts();
			const auto& seconds = sew->seconds();
			float fLen = 0, sLen = 0;
			bool invalid_sewing = false;
			for (const auto& f : firsts)
			{
				auto shape = (const ldp::AbstractShape*)Sewing::getPtrById(f.id);
				fLen += shape->getLength();
				if (m_sampleParams.find(shape) == m_sampleParams.end())
				{
					printf("warning: invalid sewing %d\n", shape->getId());
					invalid_sewing = true;
					break;
				}
			} // end for f
			for (const auto& s : seconds)
			{
				auto shape = (const ldp::AbstractShape*)Sewing::getPtrById(s.id);
				sLen += shape->getLength();
				if (m_sampleParams.find(shape) == m_sampleParams.end())
				{
					printf("warning: invalid sewing %d\n", shape->getId());
					invalid_sewing = true;
					break;
				}
			} // end for f

			if (invalid_sewing)
				continue;

			// make the step in the same range
			float fStep = step, sStep = step;
			if (fLen < sLen)
				fStep *= fLen / sLen;
			else
				sStep *= sLen / fLen;
			for (auto& f : firsts)
			{
				auto shape = (const AbstractShape*)Sewing::getPtrById(f.id);
				addSampleParam(shape, fStep / shape->getLength());
			}
			for (auto& s : seconds)
			{
				auto shape = (const AbstractShape*)Sewing::getPtrById(s.id);
				addSampleParam(shape, sStep / shape->getLength());
			}
			sew_valid[isew] = 1;
		} // end for sew

		//////////////////////////////////////////////////////////////////////
		/// until this step, we can make the triangulation density similar w.r.t the sewing
		//////////////////////////////////////////////////////////////////////
		return;

		// 3. consider param matching: if sewing.first[t] exist in sewing.second[t]
		std::vector<float> fLens, sLens;
		std::vector<SampleParam> tmpF, tmpS;
		for (size_t isew = 0; isew < m_sewings->size(); isew++)
		{
			if (!sew_valid[isew])
				continue;
			const auto& sew = (*m_sewings)[isew];
			const auto& firsts = sew->firsts();
			const auto& seconds = sew->seconds();
			fLens.clear();
			sLens.clear();
			fLens.push_back(0.f);
			sLens.push_back(0.f);
			for (const auto& f : firsts)
			{
				auto shape = (const ldp::AbstractShape*)Sewing::getPtrById(f.id);
				fLens.push_back(fLens.back() + shape->getLength());
			} // end for f
			for (const auto& s : seconds)
			{
				auto shape = (const ldp::AbstractShape*)Sewing::getPtrById(s.id);
				sLens.push_back(sLens.back() + shape->getLength());
			} // end for f

			// normalize
			for (auto& l : fLens)
				l /= fLens.back();
			for (auto& l : sLens)
				l /= sLens.back();

			// make param conincident
			size_t fPos = 0, sPos = 0, fPos_1=0, sPos_1=0;
			for (; fPos < firsts.size() && sPos < seconds.size();)
			{
				const auto& f = firsts[fPos];
				const auto& s = seconds[sPos];
				auto fshape = (const AbstractShape*)Sewing::getPtrById(f.id);
				auto sshape = (const AbstractShape*)Sewing::getPtrById(s.id);
				auto& fparams = m_sampleParams[fshape]->params;
				auto& sparams = m_sampleParams[sshape]->params;
				const float fStart = fLens[fPos];
				const float fLen = fLens[fPos + 1] - fStart;
				const float sStart = sLens[sPos];
				const float sLen = sLens[sPos + 1] - sStart;

				// reverse before matching
				//if (f.reverse)
				//	std::reverse(fparams.begin(), fparams.end());
				//if (s.reverse)
				//	std::reverse(sparams.begin(), sparams.end());

				// matching now
				for (; fPos_1 < fparams.size() && sPos_1 < sparams.size();)
				{
					float ft = f.reverse ? 1 -1+ fparams[fPos_1].t : fparams[fPos_1].t;
					float st = s.reverse ? 1 -1+ sparams[sPos_1].t : sparams[sPos_1].t;
					ft = ft * fLen + fStart;
					st = st * sLen + sStart;
					if (fabs(ft - st) < thre)
					{
						tmpF.push_back(fparams[fPos_1++]);
						tmpS.push_back(sparams[sPos_1++]);
					}
					else if (ft < st)
					{
						tmpS.push_back(SampleParam((ft-sStart)/sLen, 0));
						fPos_1++;
					}
					else
					{
						tmpF.push_back(SampleParam((st - fStart) / fLen, 0));
						sPos_1++;
					}
				} // end for fPos_1, sPos_1

				if (fPos_1 == fparams.size())
				{
					fparams = tmpF;
					tmpF.clear();
					fPos++;
					fPos_1 = 0;
					//if (f.reverse)
					//	std::reverse(fparams.begin(), fparams.end());
				}
				if (sPos_1 == sparams.size())
				{
					sparams = tmpS;
					tmpS.clear();
					sPos++;
					sPos_1 = 0;
					//if (s.reverse)
					//	std::reverse(sparams.begin(), sparams.end());
				}

				// recover the reversing
			} // end for fpos, spos
		} // end for sew
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

