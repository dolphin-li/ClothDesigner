#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "GraphLine.h"
#include "GraphQuadratic.h"
#include "GraphCubic.h"
#include "GraphLoop.h"
#include "GraphsSewing.h"
#include "tinyxml\tinyxml.h"
#include "ldpMat\ldp_basic_mat.h"
#include "cloth\definations.h"
#include <eigen\Dense>
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
		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			float dif = pointSegDistance(pts[i], pts[0], pts.back());
			if (dif > err)
			{
				err = dif;
				pos = i;
			}
		}

		line.resize(2);
		line[0].reset(new GraphPoint(pts[0]));
		line[1].reset(new GraphPoint(pts.back()));

		return std::make_pair(pos, err);
	}

	static std::pair<int, float> quadraticFitting(std::vector<Float2> pts, std::vector<GraphPointPtr>& curve)
	{
		const int nInputs = (int)pts.size();

		// prevent overfitting
		std::vector<int> idxMap(pts.size(), 0);
		for (size_t i = 0; i < idxMap.size(); i++)
			idxMap[i] = i;
		float avgLength = 0;
		for (size_t i = 1; i < pts.size(); i++)
			avgLength += (pts[i] - pts[i - 1]).length();
		avgLength /= pts.size() - 1;
		std::vector<Float2> ptsInc;
		idxMap.clear();
		for (size_t i = 0; i < pts.size(); i++)
		{
			ptsInc.push_back(pts[i]);
			idxMap.push_back(i);
			if (i + 1 < pts.size())
			{
				const float totalLen = (pts[i + 1] - pts[i]).length();
				for (float t = 0; t < totalLen; t += avgLength)
				{
					ptsInc.push_back((1 - t / totalLen)*pts[i] + t / totalLen*pts[i + 1]);
					idxMap.push_back(i);
				}
			}
		}
		pts = ptsInc;

		// calculate params
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i - 1];
		for (auto& l : paramLen)
			l /= paramLen.back();

		// weighted least square fitting
		const double lambda = 1e4;
		Eigen::MatrixXd A, B;
		A.resize(pts.size() + 2, 3);
		A.setZero();
		B.resize(A.rows(), 2);
		B.setZero();
		for (int r = 0; r < (int)pts.size(); r++)
		{
			float t = paramLen[r];
			float w = 0;
			if (r > 0)
				w += t - paramLen[r - 1];
			if (r < (int)pts.size() - 1)
				w += paramLen[r + 1] - t;
			A(r, 0) = w * (1 - t)*(1 - t);
			A(r, 1) = w * 2 * (1 - t)*t;
			A(r, 2) = w * t*t;
			B(r, 0) = w * pts[r][0];
			B(r, 1) = w * pts[r][1];
		}
		A(pts.size(), 0) = lambda;
		B(pts.size(), 0) = lambda * pts[0][0];
		B(pts.size(), 1) = lambda * pts[0][1];
		A(pts.size() + 1, 2) = lambda;
		B(pts.size() + 1, 0) = lambda * pts.back()[0];
		B(pts.size() + 1, 1) = lambda * pts.back()[1];
		auto AtA = A.transpose() * A;
		auto AtB = A.transpose() * B;
		Eigen::MatrixXd X = AtA.inverse() * AtB;

		// create cruve
		Float2 bezCurve[3];
		for (int i = 0; i < 3; i++)
			bezCurve[i] = Float2(X(i, 0), X(i, 1));

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
		}

		return std::make_pair(std::min(nInputs-1, std::max(1, idxMap[pos])), err);
	}

	static std::pair<int, float> cubicFitting(std::vector<Float2> pts, std::vector<GraphPointPtr>& curve)
	{
		const int nInputs = (int)pts.size();

		// prevent overfitting
		std::vector<int> idxMap(pts.size(), 0);
		for (size_t i = 0; i < idxMap.size(); i++)
			idxMap[i] = i;
		float avgLength = 0;
		for (size_t i = 1; i < pts.size(); i++)
			avgLength += (pts[i] - pts[i - 1]).length();
		avgLength /= pts.size() - 1;
		std::vector<Float2> ptsInc;
		idxMap.clear();
		for (size_t i = 0; i < pts.size(); i++)
		{
			ptsInc.push_back(pts[i]);
			idxMap.push_back(i);
			if (i + 1 < pts.size())
			{
				const float totalLen = (pts[i + 1] - pts[i]).length();
				for (float t = 0; t < totalLen; t += avgLength)
				{
					ptsInc.push_back((1-t/totalLen)*pts[i] + t/totalLen*pts[i + 1]);
					idxMap.push_back(i);
				}
			}
		}
		pts = ptsInc;

		// compute params
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i - 1];
		for (auto& l : paramLen)
			l /= paramLen.back();

		// weighted least square fitting
		const double lambda = 1e4;
		Eigen::MatrixXd A, B;
		A.resize(pts.size() + 2, 4);
		A.setZero();
		B.resize(A.rows(), 2);
		B.setZero();
		for (int r = 0; r < (int)pts.size(); r++)
		{
			float t = paramLen[r];
			float w = 0;
			if (r > 0)
				w += t - paramLen[r - 1];
			if (r < (int)pts.size() - 1)
				w += paramLen[r + 1] - t;
			A(r, 0) = w * (1 - t)*(1 - t)*(1 - t);
			A(r, 1) = w * 3 * (1 - t) * (1 - t)*t;
			A(r, 2) = w * 3 * (1 - t) * t*t;
			A(r, 3) = w * t* t*t;
			B(r, 0) = w * pts[r][0];
			B(r, 1) = w * pts[r][1];
		}
		A(pts.size(), 0) = lambda;
		B(pts.size(), 0) = lambda * pts[0][0];
		B(pts.size(), 1) = lambda * pts[0][1];
		A(pts.size() + 1, 3) = lambda;
		B(pts.size() + 1, 0) = lambda * pts.back()[0];
		B(pts.size() + 1, 1) = lambda * pts.back()[1];
		auto AtA = A.transpose() * A;
		auto AtB = A.transpose() * B;
		Eigen::MatrixXd X = AtA.inverse() * AtB;

		// create curve
		Float2 bezCurve[4];
		for (int i = 0; i < 4; i++)
			bezCurve[i] = Float2(X(i, 0), X(i, 1));

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
		}

		return std::make_pair(std::min(nInputs - 1, std::max(1, idxMap[pos])), err);
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