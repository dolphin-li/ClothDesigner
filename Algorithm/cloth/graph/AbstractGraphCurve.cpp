#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "GraphLine.h"
#include "GraphQuadratic.h"
#include "GraphCubic.h"
#include "GraphLoop.h"
#include "GraphsSewing.h"
#include "tinyxml\tinyxml.h"
#include "ldpMat\ldp_basic_mat.h"
namespace ldp
{
	AbstractGraphCurve::AbstractGraphCurve() : AbstractGraphObject()
	{
	
	}

	AbstractGraphCurve::AbstractGraphCurve(const std::vector<GraphPoint*>& pts) : AbstractGraphCurve()
	{
		m_keyPoints = pts;
	}

	AbstractGraphCurve::~AbstractGraphCurve()
	{
		// ldp TODO: remove related sewings
		auto tmpSewings = m_sewings;
		for (auto& sew : tmpSewings)
		{
			auto tmp = sew->firsts();
			bool found = false;
			for (auto& t : tmp)
			{
				if (t.curve == this)
				{
					sew->remove(t.curve->getId());
					found = true;
					break;
				}
			} // end for t
			if (!found)
			{
				tmp = sew->seconds();
				for (auto& t : tmp)
				{
					if (t.curve == this)
					{
						sew->remove(t.curve->getId());
						found = true;
						break;
					}
				} // end for t
			} // end if found
			if (!found)
				printf("~AbstractGraphCurve warning: curve %d not relate to sew %d\n", getId(), sew->getId());
		} // end for sew
	}

	float AbstractGraphCurve::calcLength()const
	{
		for (int k = 0; k < numKeyPoints(); k++)
		{
			assert(m_keyPoints[k]);
		}
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

	TiXmlElement* AbstractGraphCurve::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);
		for (const auto& kp : m_keyPoints)
		{
			TiXmlElement* child = new TiXmlElement(kp->getTypeString().c_str());
			ele->LinkEndChild(child);
			child->SetAttribute("id", kp->getId());
		}
		return ele;
	}

	void AbstractGraphCurve::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);
		m_keyPoints.clear();
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			if (child->Value() == g_graphPoint.getTypeString())
			{
				int id = 0;
				if (!child->Attribute("id", &id))
					throw std::exception(("cannot find id for " + getTypeString()).c_str());
				auto obj = getObjByIdx_loading(id);
				if (obj->getType() != g_graphPoint.getType())
					throw std::exception("type error");
				m_keyPoints.push_back((GraphPoint*)obj);
			}
		}
	}

	AbstractGraphCurve* AbstractGraphCurve::clone()const
	{
		auto shape = (AbstractGraphCurve*)AbstractGraphObject::create(getType());
		shape->m_keyPoints = m_keyPoints;
		shape->setSelected(isSelected());
		shape->m_graphLinks = m_graphLinks;
		shape->m_sewings = m_sewings;
		return shape;
	}

	void AbstractGraphCurve::translateKeyPoint(int i, ldp::Float2 t)
	{
		m_keyPoints[i]->setPosition(m_keyPoints[i]->getPosition()+ t);
		m_invalid = true;
	}

	void AbstractGraphCurve::translate(Float2 t)
	{
		for (auto& p : m_keyPoints)
			p->setPosition(p->getPosition() + t);
		m_invalid = true;
	}
	
	void AbstractGraphCurve::rotate(const ldp::Mat2f& R)
	{
		for (auto& p : m_keyPoints)
			p->setPosition(R * p->getPosition());
		m_invalid = true;
	}
	
	void AbstractGraphCurve::rotateBy(const Mat2f& R, Float2 c)
	{
		for (auto& p : m_keyPoints)
			p->setPosition(R * (p->getPosition() - c) + c);
		m_invalid = true;
	}

	void AbstractGraphCurve::scale(Float2 s)
	{
		for (auto& p : m_keyPoints)
			p->setPosition(p->getPosition() + s);
		m_invalid = true;
	}

	void AbstractGraphCurve::scaleBy(Float2 s, Float2 c)
	{
		for (auto& p : m_keyPoints)
			p->setPosition(s*(p->getPosition() - c) + c);
		m_invalid = true;
	}
	
	void AbstractGraphCurve::transform(const ldp::Mat3f& M)
	{
		for (auto& p : m_keyPoints)
		{
			ldp::Float3 p3(p->getPosition()[0], p->getPosition()[1], 1);
			p3 = M * p3;
			p->setPosition(Float2(p3[0], p3[1])/p3[2]);
		}
		m_invalid = true;
	}

	void AbstractGraphCurve::unionBound(Float2& bmin, Float2& bmax)
	{
		for (const auto& p : m_keyPoints)
		{
			for (int k = 0; k < p->getPosition().size(); k++)
			{
				bmin[k] = std::min(bmin[k], p->getPosition()[k]);
				bmax[k] = std::max(bmax[k], p->getPosition()[k]);
			}
		}
	}

	const std::vector<Float2>& AbstractGraphCurve::samplePointsOnShape(float step)const
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
		m_length = getLength();
		m_invalid = false;
		return m_samplePoints;
	}

	AbstractGraphCurve* AbstractGraphCurve::create(const std::vector<GraphPoint*>& kpts)
	{
		switch (kpts.size())
		{
		default:
			return nullptr;
		case 2:
			return new GraphLine(kpts);
		case 3:
			return new GraphQuadratic(kpts);
		case 4:
			return new GraphCubic(kpts);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	static std::pair<int, float> lineFitting(const std::vector<Float2>& pts, std::vector<GraphPointPtr>& line)
	{
		ldp::Float2 dir = (pts.back() - pts[0]).normalize();
		ldp::Float2 ct = (pts.back() + pts[0]) * 0.5f;

		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			auto p = pts[i];
			auto p0 = (p - ct).dot(dir) * dir + ct;
			float dif = (p0 - p).length();
			if (dif > err)
			{
				err = dif;
				pos = i;
			}
			err = std::max(err, (p0 - p).length());
		}

		line.resize(2);
		line[0].reset(new GraphPoint((pts[0] - ct).dot(dir) * dir + ct));
		line[1].reset(new GraphPoint((pts.back() - ct).dot(dir) * dir + ct));

		return std::make_pair(pos, err);
	}

	// bezier params
	inline float B0(float u)
	{
		return (1 - u)*(1 - u)*(1 - u);
	}

	inline float B1(float u)
	{
		return 3 * u * (1 - u) * (1 - u);
	}

	inline float B2(float u)
	{
		return 3 * u * u * (1 - u);
	}

	inline float B3(float u)
	{
		return u*u*u;
	}

	static std::pair<int, float> quadraticFitting(const std::vector<Float2>& pts, std::vector<GraphPointPtr>& curve)
	{
		const Float2 lTan = (pts[1] - pts[0]).normalize();
		const Float2 rTan = (pts[pts.size() - 2] - pts[pts.size() - 1]).normalize();
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i - 1];
		for (auto& l : paramLen)
			l /= paramLen.back();

		ldp::Float2 bezCurve[3] = { pts[0], 0, pts.back() };
		float area = rTan.cross(lTan);
		if (area == 0.f)
			bezCurve[1] = (pts[0] + pts.back()) * 0.5f;
		else
		{
			float tr = Float2(pts[0] - pts.back()).cross(lTan) / area;
			bezCurve[1] = pts.back() + tr * rTan;
		}

		curve.resize(3);
		for (int i = 0; i < curve.size(); i++)
			curve[i].reset(new GraphPoint(bezCurve[i]));

		// calculate distance
		std::vector<GraphPoint*> tmp;
		for (auto& p : curve)
			tmp.push_back(p.get());
		GraphQuadratic qc(tmp);
		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			auto p0 = qc.getPointByParam(paramLen[i]);
			float dif = (p0 - pts[i]).length();
			if (dif > err)
			{
				err = dif;
				pos = i;
			}
			err = std::max(err, (p0 - pts[i]).length());
		}

		return std::make_pair(pos, err);
	}

	static std::pair<int, float> cubicFitting(const std::vector<Float2>& pts, std::vector<GraphPointPtr>& curve)
	{
		const Float2 lTan = (pts[1] - pts[0]).normalize();
		const Float2 rTan = (pts[pts.size() - 2] - pts[pts.size() - 1]).normalize();
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i - 1];
		for (auto& l : paramLen)
			l /= paramLen.back();

		// create C and X matrices
		Mat2f C = Mat2f().zeros();
		Float2 X = Float2(0);
		for (size_t i = 0; i < pts.size(); i++)
		{
			const auto v0 = lTan * B1(paramLen[i]);
			const auto v1 = rTan * rTan * B2(paramLen[i]);
			C(0, 0) += v0.dot(v0);
			C(1, 0) += v1.dot(v0);
			C(1, 1) += v1.dot(v1);
			C(0, 1) += v0.dot(v1);
			const auto tmp = pts[i] - B0(paramLen[i])*pts[0] - B1(paramLen[i])*pts[0] -
				B2(paramLen[i])*pts.back() - B3(paramLen[i])*pts.back();
			X[0] += v0.dot(tmp);
			X[1] += v1.dot(tmp);
		}

		// compute the determinants of C and X
		float det_C0_C1 = C.det();
		const float det_C0_X = C(0, 0) * X[1] - C(0, 1) * X[0];
		const float det_X_C1 = X[0] * C(1, 1) - X[1] * C(0, 1);
		if (det_C0_C1 == 0.f)
			det_C0_C1 = C(0, 0) * C(1, 1) * 1e-6f;
		const float alpha_l = det_X_C1 / det_C0_C1;
		const float alpha_r = det_C0_X / det_C0_C1;

		ldp::Float2 bezCurve[4];
		//If alpha negative, use the Wu/Barsky heuristic (see text)
		//(if alpha is 0, you get coincident control points that lead to
		//divide by zero in any subsequent NewtonRaphsonRootFind() call.
		if (alpha_l < 1.0e-6 || alpha_r < 1.0e-6)
		{
			float dist = (pts[0] - pts.back()).length() / 3.f;

			bezCurve[0] = pts[0];
			bezCurve[3] = pts.back();
			bezCurve[1] = bezCurve[0] + lTan * dist;
			bezCurve[2] = bezCurve[3] + rTan * dist;
		}
		else
		{
			//  First and last control points of the Bezier curve are
			//  positioned exactly at the first and last data points
			//  Control points 1 and 2 are positioned an alpha distance out
			//  on the tangent vectors, left and right, respectively
			bezCurve[0] = pts[0];
			bezCurve[3] = pts.back();
			bezCurve[1] = bezCurve[0] + lTan * alpha_l;
			bezCurve[2] = bezCurve[3] + rTan * alpha_r;
		}


		curve.resize(4);
		for (int i = 0; i < curve.size(); i++)
			curve[i].reset(new GraphPoint(bezCurve[i]));

		// calculate distance
		std::vector<GraphPoint*> tmp;
		for (auto& p : curve)
			tmp.push_back(p.get());
		GraphCubic qc(tmp);
		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			auto p0 = qc.getPointByParam(paramLen[i]);
			float dif = (p0 - pts[i]).length();
			if (dif > err)
			{
				err = dif;
				pos = i;
			}
			err = std::max(err, (p0 - pts[i]).length());
		}

		return std::make_pair(pos, err);
	}

	void AbstractGraphCurve::fittingOneCurve(std::vector<GraphPointPtr>& curve,
		const std::vector<Float2>& keyPoints, float thre)
	{
		curve.clear();
		if (keyPoints.size() == 2)
		{
			for (const auto& p : keyPoints)
				curve.push_back(GraphPointPtr(new GraphPoint(p)));
			return;
		}

		// 1. fitting line
		auto err = lineFitting(keyPoints, curve);
		if (err.second < thre)
			return;

		// 2. quadratic fitting
		err = quadraticFitting(keyPoints, curve);
		if (err.second < thre)
			return;

		// 3. cubic fitting
		err = cubicFitting(keyPoints, curve);
		if (err.second < thre)
			return;
	}

	void AbstractGraphCurve::fittingCurves(std::vector<std::vector<GraphPointPtr>>& curves,
		const std::vector<Float2>& keyPoints, float thre)
	{
		assert(keyPoints.size() >= 2);

		if (keyPoints.size() == 2)
		{
			curves.push_back(std::vector<GraphPointPtr>());
			for (const auto& p : keyPoints)
				curves.back().push_back(GraphPointPtr(new GraphPoint(p)));
			return;
		}

		std::vector<GraphPointPtr> curve;

		// 1. fitting line
		auto err = lineFitting(keyPoints, curve);
		if (err.second < thre)
			return curves.push_back(curve);

		// 2. quadratic fitting
		err = quadraticFitting(keyPoints, curve);
		if (err.second < thre)
			return curves.push_back(curve);

		// 3. cubic fitting
		err = cubicFitting(keyPoints, curve);
		if (err.second < thre)
			return curves.push_back(curve);

		// 4. recursive fitting
		std::vector<Float2> left(keyPoints.begin(), keyPoints.begin() + err.first + 1);
		std::vector<Float2> right(keyPoints.begin() + err.first, keyPoints.end());
		fittingCurves(curves, left, thre);
		fittingCurves(curves, right, thre);
	}

	////////////////////////////////////////////////////////////////////////////////

	AbstractGraphCurve*& AbstractGraphCurve::DiskLinkIter::loopStartEdge() 
	{ 
		return m_lkIter->second.loop->m_startEdge; 
	}
	const AbstractGraphCurve*const& AbstractGraphCurve::DiskLinkIter::loopStartEdge()const 
	{ 
		return m_lkIter->second.loop->m_startEdge; 
	}
}