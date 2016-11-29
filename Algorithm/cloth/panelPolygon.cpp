#include "panelPolygon.h"
#include <set>
namespace ldp
{

	AbstractShape* AbstractShape::create(Type type)
	{
		switch (type)
		{
		case ldp::AbstractShape::TypeLine:
			return new Line();
		case ldp::AbstractShape::TypeQuadratic:
			return new Quadratic();
		case ldp::AbstractShape::TypeCubic:
			return new Cubic();
		case ldp::AbstractShape::TypeGeneralCurve:
			return new GeneralCurve();
		default:
			assert(0);
			return nullptr;
		}
	}

	AbstractShape* AbstractShape::create(const std::vector<Float2>& keyPoints)
	{
		std::vector<KeyPoint> pts;
		for (const auto& p : keyPoints)
		{
			KeyPoint kp;
			kp.position = p;
			pts.push_back(kp);
		}
		switch (keyPoints.size())
		{
		case 0:
		case 1:
			assert(0);
			return nullptr;
		case 2:
			return new Line(pts);
		case 3:
			return new Quadratic(pts);
		case 4:
			return new Cubic(pts);
		default:
			return new GeneralCurve(pts);
		}
	}

	AbstractShape* AbstractShape::clone()const
	{
		auto shape = create(getType());
		shape->m_keyPoints = m_keyPoints;
		for (auto p : shape->m_keyPoints)
			p.reset(p->clone());
		shape->m_idxStart = m_idxStart;
		shape->m_selected = m_selected;
		shape->m_highlighted = m_highlighted;
		return shape;
	}

	AbstractShape* GeneralCurve::clone()const
	{
		auto shape = (GeneralCurve*)AbstractShape::clone();
		shape->m_params = m_params;
		return shape;
	}

	const std::vector<Float2>& AbstractShape::samplePointsOnShape(float step)const
	{
		if (m_lastSampleStep != step)
		{
			m_invalid = true;
			m_lastSampleStep = step;
		}
		if (!m_invalid)
			return m_samplePoints;
		m_samplePoints.clear();
		for (float t = 0; t < 1 + step - 1e-8; t += step)
		{
			float t1 = std::min(1.f, t);
			m_samplePoints.push_back(getPointByParam(t1));
		}
		m_invalid = false;
		return m_samplePoints;
	}

	GeneralCurve::GeneralCurve(const std::vector<KeyPoint>& keyPoints) : AbstractShape(keyPoints)
	{
		m_params.resize(m_keyPoints.size(), 0);
		float totalLen = 0;
		for (int i = 0; i < (int)m_keyPoints.size() - 1; i++)
		{
			float len = (m_keyPoints[i + 1]->position - m_keyPoints[i]->position).length();
			m_params[i + 1] = m_params[i] + len;
			totalLen += len;
		}
		for (auto& p : m_params)
			p /= totalLen;
	}

	Float2 GeneralCurve::getPointByParam(float t)const
	{
		t = std::min(1.f, std::max(0.f, t));

		int bg = 0, ed = (int)m_params.size() - 1;
		int id = 0;
		for (; bg < ed;)
		{
			id = (bg + ed) / 2;
			if (t <= m_params[id])
			{
				ed = id;
			}
			else
			{
				if (t <= m_params[id + 1])
					break;
				else
					bg = id;
			}
		} // end for binary search
		assert(m_params[id] <= t && t <= m_params[id + 1]);
		float w1 = t - m_params[id];
		float w2 = m_params[id + 1] - t;
		return (m_keyPoints[id]->position * w2 + m_keyPoints[id + 1]->position * w1) / (w1 + w2);
	}

	float GeneralCurve::calcLength()const
	{
		float len = 0;
		for (size_t i = 1; i < m_keyPoints.size(); i++)
			len += (m_keyPoints[i]->position - m_keyPoints[i - 1]->position).length();
		return len;
	}

	float AbstractShape::calcLength()const
	{
		float t = 0, step = 0.1f;
		float len = 0.f;
		ldp::Float2 lastp = getPointByParam(t);
		for (float t = step; t <= 1 + step - 1e-3f; t += step)
		{
			float t1 = std::min(t, 1.f);
			ldp::Float2 p = getPointByParam(t1);
			len += (p - lastp).length();
			lastp = p;
		}
		return len;
	}

	PanelPolygon::PanelPolygon() : AbstractPanelObject()
	{
		m_bbox[0] = FLT_MAX;
		m_bbox[1] = -FLT_MAX;
	}

	PanelPolygon::~PanelPolygon()
	{
	
	}

	void PanelPolygon::clear()
	{
		m_outerPoly.clear();
		m_darts.clear();
		m_innerLines.clear();
		m_idxStart = 0;
	}

	void PanelPolygon::create(const Polygon& outerPoly, int idxStart)
	{
		outerPoly.cloneTo(m_outerPoly);
		updateIndex(idxStart);
	}

	void PanelPolygon::addDart(Dart& dart)
	{
		m_darts.push_back(Dart());
		dart.cloneTo(m_darts.back());
		updateIndex(m_idxStart);
	}

	void PanelPolygon::addInnerLine(std::shared_ptr<AbstractShape> line)
	{
		m_innerLines.push_back(ShapePtr(line->clone()));
		updateIndex(m_idxStart);
	}

	void PanelPolygon::updateIndex(int idx)
	{
		m_idxStart = idx;
		idx++;	//self
		m_outerPoly.updateIndex(idx);
		idx = m_outerPoly.getIdxEnd();
		for (auto& dart : m_darts)
		{
			dart.updateIndex(idx);
			idx = dart.getIdxEnd();
		}
		for (auto& ln : m_innerLines)
		{
			ln->setIdxBegin(idx);
			idx = ln->getIdxEnd();
		}
	}

	int PanelPolygon::getIdxEnd()const
	{
		if (m_innerLines.size())
			return m_innerLines.back()->getIdxEnd();
		if (m_darts.size())
			return m_darts.back().getIdxEnd();
		return m_outerPoly.getIdxEnd();
	}

	void PanelPolygon::updateBound(Float2& bmin, Float2& bmax)
	{
		m_bbox[0] = FLT_MAX;
		m_bbox[1] = FLT_MIN;
		m_outerPoly.updateBound(m_bbox[0], m_bbox[1]);
		for (auto& dart : m_darts)
			dart.updateBound(m_bbox[0], m_bbox[1]);
		for (auto& ln : m_innerLines)
			ln->unionBound(m_bbox[0], m_bbox[1]);
		for (int k = 0; k < bmin.size(); k++)
		{
			bmin[k] = std::min(bmin[k], m_bbox[0][k]);
			bmax[k] = std::max(bmax[k], m_bbox[1][k]);
		}
	}

	void PanelPolygon::select(int idx, SelectOp op)
	{
		m_tmpbufferObj.clear();
		collectObject(m_tmpbufferObj);
		for (auto obj : m_tmpbufferObj)
		{
			switch (op)
			{
			case ldp::AbstractPanelObject::SelectThis:
				obj->setSelected(idx == obj->getIdxBegin());
				break;
			case ldp::AbstractPanelObject::SelectUnion:
				if (idx == obj->getIdxBegin())
					obj->setSelected(idx == obj->getIdxBegin());
				break;
			case ldp::AbstractPanelObject::SelectAll:
				obj->setSelected(true);
				break;
			case ldp::AbstractPanelObject::SelectNone:
				obj->setSelected(false);
				break;
			case ldp::AbstractPanelObject::SelectInverse:
				obj->setSelected(!obj->isSelected());
				break;
			default:
				break;
			}
		}
	}

	void PanelPolygon::select(const std::vector<int>& indices, SelectOp op)
	{
		std::set<int> idxSet;
		for (auto idx : indices)
			idxSet.insert(idx);
		m_tmpbufferObj.clear();
		collectObject(m_tmpbufferObj);
		for (auto obj : m_tmpbufferObj)
		{
			switch (op)
			{
			case ldp::AbstractPanelObject::SelectThis:
				if (idxSet.find(obj->getIdxBegin()) != idxSet.end())
					obj->setSelected(true);
				else
					obj->setSelected(false);
				break;
			case ldp::AbstractPanelObject::SelectUnion:
				if (idxSet.find(obj->getIdxBegin()) != idxSet.end())
					obj->setSelected(true);
				break;
			case ldp::AbstractPanelObject::SelectAll:
				obj->setSelected(true);
				break;
			case ldp::AbstractPanelObject::SelectNone:
				obj->setSelected(false);
				break;
			case ldp::AbstractPanelObject::SelectInverse:
				obj->setSelected(!obj->isSelected());
				break;
			default:
				break;
			}
		}
	}

	void PanelPolygon::highLight(int idx)
	{
		m_tmpbufferObj.clear();
		collectObject(m_tmpbufferObj);
		for (auto obj : m_tmpbufferObj)
			obj->setHighlighted(idx == obj->getIdxBegin());
	}

	void PanelPolygon::collectObject(std::vector<AbstractPanelObject*>& objs)
	{
		objs.push_back(this);
		m_outerPoly.collectObject(objs);
		for (auto& dart : m_darts)
			dart.collectObject(objs);
		for (auto& sp : m_innerLines)
			sp->collectObject(objs);
	}
}