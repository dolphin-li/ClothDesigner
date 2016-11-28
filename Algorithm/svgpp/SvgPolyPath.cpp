#include "GL\glew.h"
#include "SvgPolyPath.h"
#include "SvgAttribute.h"
#include "SvgGroup.h"
#include "kdtree\PointTree.h"
namespace svg
{
#undef min
#undef max

	const static float CORNER_COS_ANGLE = cos(30.f * ldp::PI_S / 180.f);

	bool SvgEdgeGroup::intersect(const SvgEdgeGroup& rhs)
	{
		for (auto g : group)
		{
			if (rhs.group.find(g) != rhs.group.end())
				return true;
		}
		return false;
	}
	void SvgEdgeGroup::mergeWith(const SvgEdgeGroup& rhs)
	{
		for (auto g : rhs.group)
		{
			g.first->edgeGroups().insert(this);
			group.insert(g);
		}
	}

	TiXmlElement* SvgEdgeGroup::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = new TiXmlElement("ldp_poly_group");
		parent->LinkEndChild(ele);
		std::string cl = std::to_string(color[0]) + " " + std::to_string(color[1])
			+ " " + std::to_string(color[2]);
		std::string cmd;
		for (auto g : group)
		{
			cmd += std::to_string(g.first->getId());
			cmd += " ";
			cmd += std::to_string(g.second);
			cmd += " ";
		}
		cmd = cmd.substr(0, (int)cmd.size() - 1);
		ele->SetAttribute("color", cl.c_str());
		ele->SetAttribute("cmd", cmd.c_str());
		return ele;

	}

	SvgPolyPath::SvgPolyPath() :SvgPath()
	{
		m_3dCenter = 0.f;
		m_3dRot.setIdentity();
		m_cylinderDir = false;
	}

	SvgPolyPath::SvgPolyPath(int id) : SvgPath()
	{
		m_id = id;
		m_3dCenter = 0.f;
		m_3dRot.setIdentity();
		m_cylinderDir = false;
	}

	SvgPolyPath::~SvgPolyPath()
	{}

	int SvgPolyPath::numId()const
	{
		return numCorners() + numCornerEdges();
	}

	void SvgPolyPath::setSelected(bool s, int idx)
	{
		SvgPath::setSelected(s, idx);
		if (idx == -1)
			m_selectedCorner_arrayIds.clear();
		idx -= m_id;
		if (idx < 0 || idx >= numId())
			return;
		if (s)
			m_selectedCorner_arrayIds.insert(idx);
		else
			m_selectedCorner_arrayIds.clear();
	}

	void SvgPolyPath::setHighlighted(bool s, int idx)
	{
		SvgPath::setHighlighted(s, idx);
		if (idx == -1)
			m_highlightedCorner_arrayIds.clear();
		idx -= m_id;
		if (idx < 0 || idx >= numId())
			return;
		if (s)
			m_highlightedCorner_arrayIds.insert(idx);
		else
			m_highlightedCorner_arrayIds.clear();
	}

	void SvgPolyPath::render()
	{
		assert(m_gl_path_res->id);

		bool ancestorSelected = false;
		if (ancestorAfterRoot())
			ancestorSelected = ancestorAfterRoot()->isSelected();
		//if (isSelected())
		//	renderBounds(false);
		if (m_invalid)
		{
			cacheNvPaths();
			m_invalid = false;
		}// end if invalid

		configNvParams();
		auto sEdges = selectedEdgeIds();
		auto hEdges = highlighedEdgeIds();
		for (int i = 0; i < (int)m_edgeGLIds.size(); i++)
		{
			glColor3fv(attribute()->m_color.ptr());
			if (isHighlighted() || isSelected() || ancestorSelected)
				glColor3f(0, 0, 1);
			if (m_edgeGroups.size())
			{
				for (auto g : m_edgeGroups)
				{
					auto iter = g->group.find(std::make_pair(this, i));
					if (iter != g->group.end())
						glColor3fv(g->color.ptr());
				}
			}
			if (sEdges.find(i) != sEdges.end())
				glColor3f(1, 0, 0);
			if (hEdges.find(i) != hEdges.end())
				glColor3f(1, 0, 0);
			glStencilStrokePathNV(m_edgeGLIds[i]->id, 1, ~0);
			glCoverStrokePathNV(m_edgeGLIds[i]->id, GL_BOUNDING_BOX_NV);
		}
		renderSelection(false);
	}

	void SvgPolyPath::renderId()
	{
		assert(m_gl_path_res->id);
		if (m_invalid)
		{
			cacheNvPaths();
			m_invalid = false;
		}// end if invalid

		configNvParams();
		int id = globalIdFromEdgeId(0);
		for (auto glRes : m_edgeGLIds)
		{
			glColor4fv(color_from_index(id++).ptr());
			glStencilStrokePathNV(glRes->id, 1, ~0);
			glCoverStrokePathNV(glRes->id, GL_BOUNDING_BOX_NV);
		}
		renderSelection(true);
	}

	void SvgPolyPath::renderSelection(bool idxMode)
	{
		int nVerts = numCorners();
		int nEdges = numCornerEdges();
		bool ancestorSelected = false;
		if (ancestorAfterRoot())
			ancestorSelected = ancestorAfterRoot()->isSelected();
		if (isHighlighted() || isSelected() || ancestorSelected)
		{
			float sz = m_pathStyle.stroke_width;
			if (isHighlighted() && (isSelected() || ancestorSelected))
				sz *= 2;
			// render control points
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			glDisable(GL_STENCIL_TEST);

			// draw edges of quads only when not idx mode
			if (!idxMode)
			{
				glColor4f(0, 0, 1, 1);
				glLineWidth(2);
				glBegin(GL_LINES);
				for (size_t i = 0; i < nVerts; i++)
				{
					ldp::Float2 c = getCorner(i);
					glVertex2f(c[0] - sz, c[1] - sz);
					glVertex2f(c[0] + sz, c[1] - sz);

					glVertex2f(c[0] - sz, c[1] - sz);
					glVertex2f(c[0] - sz, c[1] + sz);

					glVertex2f(c[0] + sz, c[1] + sz);
					glVertex2f(c[0] + sz, c[1] - sz);

					glVertex2f(c[0] + sz, c[1] + sz);
					glVertex2f(c[0] - sz, c[1] + sz);
				}
				glEnd();
			}
			glBegin(GL_QUADS);
			auto sCorners = selectedCornerIds();
			auto hCorners = highlighedCornerIds();
			int id = globalIdFromCornerId(0);
			for (size_t i = 0; i < nVerts; i++)
			{
				if (idxMode)
					glColor4fv(color_from_index(id++).ptr());
				else
				{
					glColor4f(1, 1, 1, 1);
					if (hCorners.find(i) != hCorners.end() || sCorners.find(i) != sCorners.end())
						glColor4f(1, 0, 0, 1);
				}
				ldp::Float2 c = getCorner(i);
				glVertex2f(c[0] - sz, c[1] - sz);
				glVertex2f(c[0] + sz, c[1] - sz);
				glVertex2f(c[0] + sz, c[1] + sz);
				glVertex2f(c[0] - sz, c[1] + sz);
			}
			glEnd();
			glPopAttrib();
		}
	}

	void SvgPolyPath::cacheNvPaths()
	{
		updateEdgeRenderData();
		for (size_t i = 0; i < m_edgeGLIds.size(); i++)
		{
			glPathCommandsNV(m_edgeGLIds[i]->id,
				GLsizei(m_edgeCmds[i].size()), m_edgeCmds[i].data(),
				GLsizei(m_edgeCoords[i].size()), GL_FLOAT, m_edgeCoords[i].data());
		}
	}

	void SvgPolyPath::configNvParams()
	{
		int id = 0;
		auto sEdges = selectedEdgeIds();
		auto hEdges = highlighedEdgeIds();
		for (auto glRes : m_edgeGLIds)
		{
			float ss = 1 + (sEdges.find(id) != sEdges.end()) + (hEdges.find(id) != hEdges.end());
			glPathParameteriNV(glRes->id, GL_PATH_JOIN_STYLE_NV, lineJoinConverter(this));
			glPathParameteriNV(glRes->id, GL_PATH_END_CAPS_NV, lineCapConverter(this));
			glPathParameterfNV(glRes->id, GL_PATH_STROKE_WIDTH_NV, m_pathStyle.stroke_width * ss);
			glPathParameterfNV(glRes->id, GL_PATH_MITER_LIMIT_NV, m_pathStyle.miter_limit);
			if (m_pathStyle.dash_array.size())
			{
				glPathDashArrayNV(glRes->id, GLsizei(m_pathStyle.dash_array.size()), &m_pathStyle.dash_array[0]);
				glPathParameteriNV(glRes->id, GL_PATH_DASH_CAPS_NV, lineCapConverter(this));
				glPathParameterfNV(glRes->id, GL_PATH_DASH_OFFSET_NV, m_pathStyle.dash_offset);
				glPathParameteriNV(glRes->id, GL_PATH_DASH_OFFSET_RESET_NV, m_pathStyle.dash_phase);
			}
			else
			{
				glPathDashArrayNV(glRes->id, 0, NULL);
			}
			id++;
		}
	}

	void SvgPolyPath::copyTo(SvgAbstractObject* obj)const
	{
		SvgPath::copyTo(obj);
		if (obj->objectType() == SvgAbstractObject::PolyPath)
		{
			auto newTptr = (SvgPolyPath*)obj;
			newTptr->m_cornerPos = m_cornerPos;
			newTptr->m_highlightedCorner_arrayIds = m_highlightedCorner_arrayIds;
			newTptr->m_selectedCorner_arrayIds = m_selectedCorner_arrayIds;
			newTptr->m_edgeCmds = m_edgeCmds;
			newTptr->m_edgeCoords = m_edgeCoords;
			newTptr->m_edgeGLIds = m_edgeGLIds;
			newTptr->m_edgeGroups = m_edgeGroups;
			newTptr->m_3dCenter = m_3dCenter;
			newTptr->m_3dRot = m_3dRot;
		}
	}

	std::shared_ptr<SvgAbstractObject> SvgPolyPath::clone(bool selectedOnly)const
	{
		if (selectedOnly)
		{
			if (!(hasSelectedChildren() || isSelected()))
				throw std::exception("ERROR: SvgPolyPath::clone(), mis-called");
		}
		std::shared_ptr<SvgAbstractObject> newT(new SvgPolyPath());
		auto newTptr = (SvgPolyPath*)newT.get();

		copyTo(newTptr);

		// edgeGroups are not clonable
		newTptr->m_edgeGroups.clear();

		return newT;
	}

	std::shared_ptr<SvgAbstractObject> SvgPolyPath::deepclone(bool selectedOnly)const
	{
		if (selectedOnly)
		{
			if (!(hasSelectedChildren() || isSelected()))
				throw std::exception("ERROR: SvgPolyPath::clone(), mis-called");
		}
		std::shared_ptr<SvgAbstractObject> newT(new SvgPolyPath());
		auto newTptr = (SvgPolyPath*)newT.get();

		copyTo(newTptr);

		newTptr->m_gl_path_res.reset(new GLPathResource());
		newTptr->m_edgeGLIds.clear();

		// edgeGroups are not clonable
		newTptr->m_edgeGroups.clear();
		newTptr->invalid();
		return newT;
	}

	TiXmlElement* SvgPolyPath::toXML(TiXmlNode* parent)const
	{
		std::string cmdStr;
		char buffer[1014];
		std::vector<int> cmdPos;
		cmdPos.push_back(0);
		for (auto c : m_cmds)
			cmdPos.push_back(cmdPos.back() + numCoords(c));
		for (size_t i_cmd = 0; i_cmd < m_cmds.size(); i_cmd++)
		{
			auto str = svgCmd(m_cmds[i_cmd]);
			if (str == 'A')
				throw std::exception("arc export not implemented!");
			cmdStr += str;
			int bg = cmdPos[i_cmd], ed = cmdPos[i_cmd + 1];
			for (int i = bg; i < ed; i++)
			{
				sprintf_s(buffer, "%.2f", m_coords[i]);
				cmdStr += buffer;
				if (i < ed - 1)
					cmdStr += ',';
			}
		}

		TiXmlElement* ele = new TiXmlElement("path");
		parent->LinkEndChild(ele);
		ele->SetAttribute("fill", strokeFillMap(m_pathStyle.fill_rule));
		ele->SetAttribute("ldp_poly", m_id);
		ele->SetDoubleAttribute("ldp_3dx", m_3dCenter[0]);
		ele->SetDoubleAttribute("ldp_3dy", m_3dCenter[1]);
		ele->SetDoubleAttribute("ldp_3dz", m_3dCenter[2]);
		ele->SetDoubleAttribute("ldp_3drx", m_3dRot.v[0]);
		ele->SetDoubleAttribute("ldp_3dry", m_3dRot.v[1]);
		ele->SetDoubleAttribute("ldp_3drz", m_3dRot.v[2]);
		ele->SetDoubleAttribute("ldp_3drw", m_3dRot.w);
		ele->SetAttribute("ldp_cylinder_dir", m_cylinderDir);
		ele->SetAttribute("stroke", "#231F20");
		ele->SetDoubleAttribute("stroke-width", m_pathStyle.stroke_width);
		ele->SetAttribute("stroke-linecap", strokeLineCapMap(m_pathStyle.line_cap));
		ele->SetAttribute("stroke-linejoin", strokeLineJoinMap(m_pathStyle.line_join));
		ele->SetDoubleAttribute("stroke-miterlimit", m_pathStyle.miter_limit);
		ele->SetAttribute("d", cmdStr.c_str());
		std::string cornerPosStr;
		for (size_t i = 0; i < m_cornerPos.size(); i++)
			cornerPosStr += std::to_string(m_cornerPos[i]) + " ";
		if (cornerPosStr.size())
			cornerPosStr = cornerPosStr.substr(0, cornerPosStr.size() - 1);
		ele->SetAttribute("ldp_corner", cornerPosStr.c_str());
		return ele;
	}

	bool SvgPolyPath::isClosed()const
	{
		if (m_coords.size() < 4)
			return false;
		return m_coords[0] == m_coords[m_coords.size() - 2] &&
			m_coords[1] == m_coords[m_coords.size() - 1];
	}

	void SvgPolyPath::makeClosed()
	{
		if (isClosed()) return;
		m_cmds.push_back(GL_LINE_TO_NV);
		m_coords.insert(m_coords.end(), m_coords.begin(), m_coords.begin() + 2);
		findCorners();
		invalid();
	}

	void SvgPolyPath::findCorners()
	{
		m_cornerPos.clear();
		if (m_cmds.size() == 0)
			return;
		const bool closed = isClosed();
		if (!closed)
			m_cornerPos.push_back(0);
		for (int icmd = !closed; icmd < (int)m_cmds.size() - 1; icmd++)
		{
			int lasti = (icmd - 1 + int(m_cmds.size())) % int(m_cmds.size());
			if (closed && lasti == (int)m_cmds.size() - 1)
				lasti--;
			int nexti = icmd + 1;
			ldp::Float2 lastp(m_coords[lasti * 2], m_coords[lasti * 2 + 1]);
			ldp::Float2 p(m_coords[icmd * 2], m_coords[icmd * 2 + 1]);
			ldp::Float2 dir = (p - lastp).normalize();
			ldp::Float2 nextp(m_coords[nexti * 2], m_coords[nexti * 2 + 1]);
			ldp::Float2 ndir = (nextp - p).normalize();
			if (dir.dot(ndir) < CORNER_COS_ANGLE)
				m_cornerPos.push_back(icmd);
		} // icmd
		if (!isClosed())
			m_cornerPos.push_back((int)m_cmds.size() - 1);

		invalid();
	}

	void SvgPolyPath::bilateralSmooth(double thre)
	{
		if (m_coords.size() <= 4)
			return;
		bool closed = isClosed();

		// compute the average edge length
		double avgLength = 0;
		int nE = 0;
		for (size_t i = 2; i < m_coords.size(); i += 2)
		{
			ldp::Float2 p1(m_coords[i], m_coords[i + 1]);
			ldp::Float2 p2(m_coords[i - 2], m_coords[i - 1]);
			avgLength += (p1 - p2).length();
			nE++;
		}
		avgLength /= nE;
		const float avgLength2 = avgLength*avgLength;

		// bilateral smoothing
		std::vector<float> curCoords = m_coords;
		for (size_t iter = 0; iter < 5; iter++)
		{
			for (size_t i = 2; i < m_coords.size()-2; i += 2)
			{
				ldp::Float2 p(m_coords[i], m_coords[i + 1]);
				ldp::Float2 pp(m_coords[i - 2], m_coords[i - 1]);
				ldp::Float2 pn(m_coords[i + 2], m_coords[i + 3]);
				float wp = exp(-(p - pp).sqrLength() / avgLength2);
				float wn = exp(-(p - pn).sqrLength() / avgLength2);
				ldp::Float2 q = (p + wp*pp + wn*pn) / (1 + wp + wn);
				curCoords[i] = q[0];
				curCoords[i + 1] = q[1];
			} // end for i
			m_coords = curCoords;
		} // end for iter

		// perform merging
		std::vector<int> mergeMap(m_cmds.size());
		for (size_t i = 0; i < mergeMap.size(); i++)
			mergeMap[i] = (int)i;
		for (size_t i = 1; i < m_cmds.size() - 1; i++)
		{
			ldp::Float2 p(m_coords[i * 2], m_coords[i * 2 + 1]);
			ldp::Float2 pp(m_coords[i * 2 - 2], m_coords[i * 2 - 1]);
			float dist = (p - pp).length();
			if (dist < avgLength * thre)
			{
				mergeMap[i] = i - 1;
				while (mergeMap[i] != mergeMap[mergeMap[i]])
					mergeMap[i] = mergeMap[mergeMap[i]];
			}
		} // enf for i
		auto tmpCmds = m_cmds;
		auto tmpCoords = m_coords;
		m_cmds.clear();
		m_coords.clear();
		for (size_t i = 0; i < mergeMap.size(); i++)
		{
			if (mergeMap[i] != i) continue;
			m_cmds.push_back(tmpCmds[i]);
			m_coords.insert(m_coords.end(), tmpCoords.begin() + i * 2, tmpCoords.begin() + i * 2 + 2);
		}

		// after smoothing, we findCorners again
		auto oldCornerPos = m_cornerPos;
		findCorners();
		bool changed = false;
		if (oldCornerPos.size() != m_cornerPos.size())
			changed = true;
		if (!changed)
		{
			for (size_t i = 0; i < oldCornerPos.size(); i++)
			if (oldCornerPos[i] != m_cornerPos[i])
			{
				changed = true;
				break;
			}
		}
		if (changed)
		{
			m_selectedCorner_arrayIds.clear();
			m_highlightedCorner_arrayIds.clear();
			for (auto& eg : m_edgeGroups)
				eg->group.clear();
			m_edgeGroups.clear();
		}
	}

	void SvgPolyPath::setCorners(std::vector<int>& cns)
	{
		m_cornerPos.clear();
		if (m_cmds.size() == 0)
			return;
		for (auto c : cns)
		{
			if (c >= 0 && c < m_cmds.size())
				m_cornerPos.push_back(c);
		}
		invalid();
	}

	void SvgPolyPath::updateEdgeRenderData()
	{
		const int nCorners = numCorners();
		const int nEdges = numCornerEdges();
		m_edgeCmds.clear();
		m_edgeCoords.clear();
		m_edgeCmds.resize(nEdges);
		m_edgeCoords.resize(nEdges);
		m_edgeGLIds.resize(nEdges, 0);

		// gl resource
		for (int iedge = 0; iedge < nEdges; iedge++)
		if (m_edgeGLIds[iedge].get() == nullptr)
			m_edgeGLIds[iedge] = std::shared_ptr<GLPathResource>(new GLPathResource());

		// path data
		for (int iedge = 0; iedge < nEdges; iedge++)
		{
			auto& edgeCmd = m_edgeCmds[iedge];
			auto& edgeCoord = m_edgeCoords[iedge];
			int cb = m_cornerPos[iedge];
			int ce = iedge + 1 < nCorners ? m_cornerPos[iedge + 1] + 1 : m_cornerPos[0] + 1;
			if (ce > cb)
			{
				edgeCmd.insert(edgeCmd.end(), m_cmds.begin() + cb, m_cmds.begin() + ce);
				edgeCoord.insert(edgeCoord.end(), m_coords.begin() + cb * 2, m_coords.begin() + ce * 2);
			}
			else
			{
				edgeCmd.insert(edgeCmd.end(), m_cmds.begin() + cb, m_cmds.end());
				edgeCoord.insert(edgeCoord.end(), m_coords.begin() + cb * 2, m_coords.end());
				edgeCmd.insert(edgeCmd.end(), m_cmds.begin(), m_cmds.begin() + ce);
				edgeCoord.insert(edgeCoord.end(), m_coords.begin(), m_coords.begin() + ce * 2);
			}
			edgeCmd[0] = GL_MOVE_TO_NV;
		} // end for iedge
	}

	bool SvgPolyPath::removeSelectedCorners()
	{
		bool re = false;
		auto cornerIds = selectedCornerIds();
		if (cornerIds.size() > 1)
			throw std::exception("Error: multi-selection removal not implemented!");
		for (auto id : cornerIds)
		if (removeCorner(id))
			re = true;
		m_selectedCorner_arrayIds.clear();
		m_highlightedCorner_arrayIds.clear();
		return re;
	}

	bool SvgPolyPath::splitSelectedEdgeMidPoint()
	{
		bool re = false;
		auto edgeIds = selectedEdgeIds();
		if (edgeIds.size() > 1)
			throw std::exception("Error: multi-selection removal not implemented!");
		for (auto id : edgeIds)
		if (splitSelectedEdgeMidPoint(id))
			re = true;
		m_selectedCorner_arrayIds.clear();
		m_highlightedCorner_arrayIds.clear();
		return re;
	}

	bool SvgPolyPath::splitSelectedEdgeMidPoint(int edge_arrayId)
	{
		if (edge_arrayId < 0 || edge_arrayId >= numCornerEdges())
			return false;

		auto tmpCmds = m_edgeCmds[edge_arrayId];
		auto tmpCoords = m_edgeCoords[edge_arrayId];
		float len = 0.f;
		for (size_t i = 2; i < tmpCoords.size(); i += 2)
			len += (ldp::Float2(tmpCoords[i], tmpCoords[i + 1]) -
			ldp::Float2(tmpCoords[i - 2], tmpCoords[i - 1])).length();
		const float halfLen = len * 0.5f;
		len = 0.f;
		ldp::Float2 midP = 0.f;
		int midIdx = -1;
		for (size_t i = 2; i < tmpCoords.size(); i += 2)
		{
			ldp::Float2 p2(tmpCoords[i], tmpCoords[i + 1]);
			ldp::Float2 p1(tmpCoords[i - 2], tmpCoords[i - 1]);
			float d12 = (p2 - p1).length();
			len += d12;
			if (len > halfLen)
			{
				float w = (len - halfLen) / d12;
				midP = p1 * (1 - w) + p2 * w;
				midIdx = i / 2 - 1;
				break;
			}
		} // end for i

		midIdx += m_cornerPos[edge_arrayId];
		midIdx %= m_cmds.size(); ///// how to handle looped corners???
		m_cmds.insert(m_cmds.begin() + midIdx + 1, GL_LINE_TO_NV);
		m_coords.insert(m_coords.begin() + (midIdx + 1) * 2, midP.ptr(), midP.ptr() + 2);
		for (size_t i = edge_arrayId + 1; i < m_cornerPos.size(); i++)
			m_cornerPos[i]++;
		m_cornerPos.insert(m_cornerPos.begin() + edge_arrayId + 1, midIdx + 1);
		updateEdgeRenderData();
		invalid();
		for (auto& eg : m_edgeGroups)
		{
			auto tmp = eg->group;
			eg->group.clear();
			for (auto g : tmp)
			{
				if (g.first == this && g.second > edge_arrayId)
					g.second++;
				eg->group.insert(g);
			} // g
		} // eg
		return true;
	}

	bool SvgPolyPath::removeCorner(int corner_arrayId)
	{
		if (corner_arrayId < 0 || corner_arrayId >= numCorners())
			return false;
		if (!isClosed())
		{
			if (corner_arrayId == 0 || corner_arrayId == numCorners() - 1)
				return false;
		}

		int lastId = corner_arrayId - 1;
		if (isClosed() && lastId < 0)
			lastId = numCorners() - 1;

		auto tmpCmds = m_edgeCmds[corner_arrayId];
		auto tmpCoords = m_edgeCoords[corner_arrayId];
		m_edgeCmds[lastId].insert(m_edgeCmds[lastId].end(), tmpCmds.begin() + 1, tmpCmds.end());
		m_edgeCoords[lastId].insert(m_edgeCoords[lastId].end(), tmpCoords.begin() + 2, tmpCoords.end());
		m_edgeCmds.erase(m_edgeCmds.begin() + corner_arrayId);
		m_edgeCoords.erase(m_edgeCoords.begin() + corner_arrayId);
		m_cornerPos.erase(m_cornerPos.begin() + corner_arrayId);
		m_edgeGLIds.erase(m_edgeGLIds.begin() + corner_arrayId);

		invalid();
		for (auto& eg : m_edgeGroups)
		{
			auto tmp = eg->group;
			eg->group.clear();
			for (auto g : tmp)
			{
				if (g.first == this && g.second == corner_arrayId)
					continue;
				if (g.first == this && g.second > corner_arrayId)
					g.second--;
				eg->group.insert(g);
			} // g
		} // eg
		return true;
	}

	float SvgPolyPath::calcAreaCornerWise()const
	{
		float area = 0.f;
		const int n = numCorners();
		for (int i = 0, j = n - 1; i < n; j = i++)
			area += getCorner(j).cross(getCorner(i));
		return area * 0.5f;
	}
}