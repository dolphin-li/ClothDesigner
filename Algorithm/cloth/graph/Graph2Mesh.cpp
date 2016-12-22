#include "Graph2Mesh.h"
#include "ldpMat\ldp_basic_mat.h"
#include "cloth\clothPiece.h"
#include "cloth\graph\Graph.h"
#include "cloth\graph\GraphsSewing.h"
#include "cloth\graph\AbstractGraphCurve.h"
#include "cloth\graph\GraphLoop.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "kdtree\PointTree.h"
extern "C"{
#include "triangle\triangle.h"
};

namespace ldp
{
	inline float calcForwardBackwardConsistentStep(float step)
	{
		step = std::max(0.f, std::min(1.f, step));
		int n = std::lroundf(1.f/step);
		return 1.f / float(n);
	}

	static int findParamRange(float t, const std::vector<float>& ranges)
	{
		int bg = 0, ed = (int)ranges.size() - 1;
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

	Graph2Mesh::Graph2Mesh()
	{
		m_in = new triangulateio;
		m_out = new triangulateio;
		m_vro = new triangulateio;
		init_trianglulateio(m_in);
		init_trianglulateio(m_out);
		init_trianglulateio(m_vro);
	}

	Graph2Mesh::~Graph2Mesh()
	{
		reset_triangle_struct(m_in);
		m_out->numberofholes = 0;
		m_out->holelist = nullptr;
		reset_triangle_struct(m_out);
		reset_triangle_struct(m_vro);
		delete m_in;
		delete m_out;
		delete m_vro;
	}

	void Graph2Mesh::triangulate(
		std::vector<std::shared_ptr<ClothPiece>>& pieces,
		std::vector<std::shared_ptr<GraphsSewing>>& sewings,
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
			piece->mesh2d().clear();
			piece->mesh3d().clear();
			piece->mesh3dInit().clear();
			const auto& panel = piece->graphPanel();
			auto bloop = panel.getBoundingLoop();
			if (bloop == nullptr)
				continue;
			prepareTriangulation(); 

			// add bounding loop as the outer poly
			addPolygon(*bloop);

			// add other closed loops as the darts
			for (auto loop_iter = panel.loop_begin(); loop_iter != panel.loop_end(); ++loop_iter)
			{
				if (loop_iter->isClosed() && loop_iter != bloop)
					addDart(*loop_iter);
			} // end for loop iter

			// add inner lines
			for (auto loop_iter = panel.loop_begin(); loop_iter != panel.loop_end(); ++loop_iter)
			{
				if (!loop_iter->isClosed())
					addLine(*loop_iter); // ldp todo: add dart?
			} // end for loop iter
			finalizeTriangulation();
			generateMesh(*piece.get());
		} // end for piece
		postComputeSewing();
	}

	void Graph2Mesh::prepareTriangulation()
	{
		reset_triangle_struct(m_in);
		m_out->numberofholes = 0;
		m_out->holelist = nullptr;
		reset_triangle_struct(m_out);
		reset_triangle_struct(m_vro);
		m_points.clear();
		m_segments.clear();
		m_holeCenters.clear();
		m_triBuffer.clear();
		m_triVertsBuffer.clear();
	}

	void Graph2Mesh::precomputeSewing()
	{
		const float step = m_triSize;
		const float thre = m_ptMergeThre;

		m_shapeSegs.clear();
		m_segPairs.clear();
		m_segStepMap.clear();

		// 1. convert each shape to a segment, without considering the sewings
		for (auto& piece : (*m_pieces))
		{
			const auto& panel = piece->graphPanel();
			for (auto iter = panel.curve_begin(); iter != panel.curve_end(); ++iter)
			{
				createShapeSeg(iter, calcForwardBackwardConsistentStep(step / iter->getLength()));
				m_shapePieceMap[iter] = piece.get();
			}
		} // end for piece

		// 2. split a shape segment to multiple based on the M-N sew matching.
		//    after this step, the sewing must be 1-to-1 correspondence
		// LDP NOTES: seems AB + AC sewing cannot be handled.
		std::vector<float> fLens, sLens;
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
				fLens.push_back(fLens.back() + f.curve->getLength());
			for (const auto& s : seconds)
				sLens.push_back(sLens.back() + s.curve->getLength());
			for (auto& f : fLens)
				f /= fLens.back();
			for (auto& s : sLens)
				s /= sLens.back();

			size_t fPos = 0, sPos = 0;
			for (; fPos+1 < fLens.size() && sPos+1 < sLens.size();)
			{
				const float fStart = fLens[fPos], fEnd = fLens[fPos + 1];
				const float sStart = sLens[sPos], sEnd = sLens[sPos + 1];
				const float fLen = fEnd - fStart, sLen = sEnd - sStart;
				const auto& fUnit = firsts[fPos];
				const auto fShape = fUnit.curve;
				auto& fSegs = m_shapeSegs[fShape];
				const auto& sUnit = seconds[sPos];
				const auto sShape = sUnit.curve;
				auto& sSegs = m_shapeSegs[sShape];
				size_t fPos_1 = fSegs->size() - 1;
				size_t sPos_1 = sSegs->size() - 1;
				if (fUnit.reverse)
					fPos_1 = 0;
				if (sUnit.reverse)
					sPos_1 = 0;
				if (fabs(fEnd - sEnd) < thre)
				{
					fPos++;
					sPos++;
				} // end if f, s too close
				else if (fEnd > sEnd)
				{
					float local = (sEnd - fStart) / fLen;
					if (fUnit.reverse)
						local = 1.f - local;
					fPos_1 = addSegToShape(*fSegs.get(), local) + fUnit.reverse;
					sPos++;
				} // end if f > s
				else
				{
					float local = (fEnd - sStart) / sLen;
					if (sUnit.reverse)
						local = 1.f - local;
					sPos_1 = addSegToShape(*sSegs.get(), local) + sUnit.reverse;
					fPos++;
				} // end else f < s
				const float minStep = calcForwardBackwardConsistentStep(
					std::min((*fSegs)[fPos_1]->step, (*sSegs)[sPos_1]->step));
				updateSegStepMap((*fSegs)[fPos_1].get(), minStep);
				updateSegStepMap((*sSegs)[sPos_1].get(), minStep);
				m_segPairs.push_back(SegPair(fSegs->at(fPos_1).get(), fUnit.reverse,
					sSegs->at(sPos_1).get(), sUnit.reverse));
			} // end for fPos, sPos
		} // end for sew

		// 3. perform resampling, such that each sewing pair share the same sample points
		for (auto& pair : m_segPairs)
		{
			for (int k = 0; k < 2; k++)
			{
				float step = m_segStepMap[pair.seg[k]];
				resampleSeg(*pair.seg[k], step);
			}
		} // end for pair
	}

	void Graph2Mesh::createShapeSeg(const AbstractGraphCurve* shape, float step)
	{
		auto iter = m_shapeSegs.find(shape);
		if (iter == m_shapeSegs.end())
		{
			ShapeSegsPtr ptr(new ShapeSegs);
			ptr->resize(1); 
			(*ptr)[0].reset(new SampleParamVec);
			(*ptr)[0]->shape = shape;
			(*ptr)[0]->start = 0;
			(*ptr)[0]->end = 1;
			(*ptr)[0]->step = step;
			for (float s = 0.f; s < 1 + step - g_designParam.pointMergeDistThre; s += step)
				(*ptr)[0]->params.push_back(SampleParam(std::min(1.f, s), -1));
			m_shapeSegs[shape] = ptr;
		} // end if iter not found
		else
			throw std::exception("duplicated shape added!\n");
	}

	int Graph2Mesh::addSegToShape(ShapeSegs& segs, float tSegs)
	{
		for (size_t i = 0; i< segs.size(); i++)
		{
			const float segBegin = segs[i]->start;
			const float segEnd = segs[i]->end;
			if (tSegs > segBegin && tSegs < segEnd)
			{
				segs.insert(segs.begin() + i + 1, SampleParamVecPtr(new SampleParamVec()));
				auto& oVec = segs[i];
				auto& sVec = segs[i + 1];
				const float oriStep = oVec->step;
				oVec->step = calcForwardBackwardConsistentStep(oriStep * (segEnd - segBegin) / (tSegs - segBegin));
				oVec->start = segBegin;
				oVec->end = tSegs;
				oVec->params.clear();
				for (float s = 0.f; s < 1 + oVec->step - 1e-8; s += oVec->step)
					oVec->params.push_back(SampleParam(std::min(1.f, s), -1));
				sVec->shape = oVec->shape;
				sVec->step = calcForwardBackwardConsistentStep(oriStep * (segEnd - segBegin) / (segEnd - tSegs));
				sVec->start = tSegs;
				sVec->end = segEnd;
				sVec->params.clear();
				for (float s = 0.f; s < 1 + sVec->step - 1e-8; s += sVec->step)
					sVec->params.push_back(SampleParam(std::min(1.f, s), -1));
				return i;
			} // end if >, <
		} // end for i
		return -1;
	}

	void Graph2Mesh::updateSegStepMap(SampleParamVec* seg, float step)
	{
		auto iter = m_segStepMap.find(seg);
		if (iter == m_segStepMap.end())
		{
			m_segStepMap.insert(std::make_pair(seg, step));
		}
		else
		{
			iter->second = std::min(step, iter->second);
		}
	}

	void Graph2Mesh::resampleSeg(SampleParamVec& seg, float step)
	{
		if (fabs(seg.step - step) < std::numeric_limits<float>::epsilon())
			return;
		seg.step = step;
		seg.params.clear();
		for (float s = 0.f; s < 1 + seg.step - g_designParam.pointMergeDistThre; s += seg.step)
			seg.params.push_back(SampleParam(std::min(1.f, s), -1));
	}

	Float2 Graph2Mesh::addPolygon(const GraphLoop& poly)
	{
		// build a kdtree for existed points
		typedef kdtree::PointTree<float, 2> Tree;
		typedef Tree::Point Point;
		std::vector<Point> treePoints;
		for (int i = 0; i < m_points.size(); i++)
			treePoints.push_back(Point(Float2(m_points[i]), i));
		Tree tree;
		tree.build(treePoints);

		// begin
		const float step = m_triSize;
		const float thre = m_ptMergeThre;
		int startIdx = (int)m_points.size();

		// add points
		std::vector<int> indices;
		Float2 center = 0.f;
		for (auto edge_iter = poly.edge_begin(); !edge_iter.isEnd(); ++edge_iter)
		{
			//TODO: reverse edge if needed
			auto& shapeSegs = m_shapeSegs[&(*edge_iter)];
			int segBegin = 0, segEnd = (int)shapeSegs->size(), segInc = 1;
			if (edge_iter.shouldReverse())
			{
				segBegin = (int)shapeSegs->size() - 1;
				segEnd = -1;
				segInc = -1;
			}
			for (int iSeg = segBegin; iSeg != segEnd; iSeg+=segInc)
			{
				auto& seg = shapeSegs->at(iSeg);
				int paramBegin = 0, paramEnd = (int)seg->params.size(), paramInc = 1;
				if (edge_iter.shouldReverse())
				{
					paramBegin = (int)seg->params.size() - 1;
					paramEnd = -1;
					paramInc = -1;
				}
				for (int iParam = paramBegin; iParam != paramEnd; iParam += paramInc)
				{
					auto& sp = seg->params[iParam];
					auto p = edge_iter->getPointByParam(sp.t * (seg->end - seg->start) + seg->start);
					// if this shape is not used, we create points for it
					// else we just use its idx
					if (sp.idx == -1)
					{
						if (m_points.size() != startIdx)
						{
							if ((m_points.back() - p).length() < thre)
								sp.idx = int(m_points.size()) - 1; // merged to the last point
							else if ((m_points[startIdx] - p).length() < thre)
								sp.idx = startIdx; // merged to the last point
						}
						if (sp.idx == -1)
						{
							// if there exists too close points, just use it.
							float dist = 0;
							auto np = tree.nearestPoint(p, dist);
							if (dist < thre)
								sp.idx = np.idx;
							else
							{
								sp.idx = int(m_points.size());
								m_points.push_back(p);
							}
						}
					} // end if sp.idx == -1
					indices.push_back(sp.idx);
					center += m_points[sp.idx];
				} // end for sp
			} // end for p
		} // end for edge_iter

		// add segments
		for (int i = 0; i < (int)indices.size() - 1; i++)
			m_segments.push_back(Int2(indices[i], indices[i+1]));
		if (indices.size() > 1 && poly.isClosed())
			m_segments.push_back(Int2(indices.back(), indices.front()));

		if (indices.size())
			center /= indices.size();
		return center;
	}

	void Graph2Mesh::addDart(const GraphLoop& poly)
	{
		Float2 center = addPolygon(poly);
		m_holeCenters.push_back(center);
	}

	void Graph2Mesh::addLine(const GraphLoop& line)
	{
		addPolygon(line);
	}


	void Graph2Mesh::finalizeTriangulation()
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
		
		// init segments, unique it before using it.
		for (auto& s : m_segments)
		{
			if (s[0] > s[1])
				std::swap(s[0], s[1]);
		}
		std::sort(m_segments.begin(), m_segments.end());
		m_segments.resize(std::unique(m_segments.begin(), m_segments.end())-m_segments.begin());
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

	void Graph2Mesh::generateMesh(ClothPiece& piece)
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

	void Graph2Mesh::postComputeSewing()
	{
		// 1. count the verts of each piece and accumulate
		m_vertStart.clear();
		int vertNum = 0;
		for (const auto& piece : *m_pieces)
		{
			m_vertStart[piece.get()] = vertNum;
			vertNum += (int)piece->mesh2d().vertex_list.size();
		}

		// 2. make stitching
		m_stitches.clear();
		for (const auto& pair : m_segPairs)
		{
			std::vector<SampleParam> param0 = pair.seg[0]->params;
			std::vector<SampleParam> param1 = pair.seg[1]->params;
			assert(param0.size() == param1.size());
			if (pair.reverse[0])
				std::reverse(param0.begin(), param0.end());
			if (pair.reverse[1])
				std::reverse(param1.begin(), param1.end());
			for (size_t i = 0; i < param0.size(); i++)
			{
				auto piece0 = m_shapePieceMap[pair.seg[0]->shape];
				auto piece1 = m_shapePieceMap[pair.seg[1]->shape];
				int s0 = m_vertStart[piece0];
				int s1 = m_vertStart[piece1];
				int id0 = param0[i].idx + s0;
				int id1 = param1[i].idx + s1;
				if (id0 == id1)
					continue;
				StitchPointPair stp;
				stp.first.vids = id0;
				stp.first.w = 0;
				stp.second.vids = id1;
				stp.second.w = 0;
				m_stitches.push_back(stp);
			}
		} // end for pair
	}

	void Graph2Mesh::reset_triangle_struct(triangulateio* io)const
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

