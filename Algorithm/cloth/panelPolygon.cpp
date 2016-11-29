#include "panelPolygon.h"

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

	AbstractShape* AbstractShape::create(const std::vector<Point>& keyPoints)
	{
		switch (keyPoints.size())
		{
		case 0:
		case 1:
			assert(0);
			return nullptr;
		case 2:
			return new Line(keyPoints);
		case 3:
			return new Quadratic(keyPoints);
		case 4:
			return new Cubic(keyPoints);
		default:
			return new GeneralCurve(keyPoints);
		}
	}

	const std::vector<AbstractShape::Point>& AbstractShape::samplePointsOnShape(float step)const
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

	GeneralCurve::GeneralCurve(const std::vector<Point>& keyPoints) : AbstractShape()
	{
		m_keyPoints = keyPoints;
		m_params.resize(m_keyPoints.size(), 0);
		float totalLen = 0;
		for (int i = 0; i < (int)m_keyPoints.size() - 1; i++)
		{
			float len = (m_keyPoints[i + 1].length() - m_keyPoints[i]).length();
			m_params[i + 1] = m_params[i] + len;
			totalLen += len;
		}
		for (auto& p : m_params)
			p /= totalLen;
	}

	GeneralCurve::Point GeneralCurve::getPointByParam(float t)const
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
		return (m_keyPoints[id] * w2 + m_keyPoints[id + 1] * w1) / (w1 + w2);
	}

	float GeneralCurve::calcLength()const
	{
		float len = 0;
		for (size_t i = 1; i < m_keyPoints.size(); i++)
			len += (m_keyPoints[i] - m_keyPoints[i - 1]).length();
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

	PanelPolygon::PanelPolygon()
	{
		m_idxStart = 0;
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

	int PanelPolygon::getIndexBegin()const
	{
		return m_idxStart;
	}

	int PanelPolygon::getIndexEnd()const
	{
		if (m_innerLines.size())
			return m_innerLines.back()->getIdxEnd();
		if (m_darts.size())
			return m_darts.back().getIdxEnd();
		return m_outerPoly.getIdxEnd();
	}

	void PanelPolygon::updateBound(Point& bmin, Point& bmax)
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
}