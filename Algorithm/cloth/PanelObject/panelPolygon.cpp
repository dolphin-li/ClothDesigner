#include "panelPolygon.h"
#include <eigen\Dense>
#include <exception>
namespace ldp
{
	std::hash_map<size_t, AbstractPanelObject*> AbstractPanelObject::s_globalIdxMap;
	bool AbstractPanelObject::s_disableIdxMapUpdate = false;
	std::hash_map<AbstractPanelObject::Type, std::string> AbstractPanelObject::generateTypeStringMap()
	{
		std::hash_map<AbstractPanelObject::Type, std::string> map;
		map.clear();
		map[TypeKeyPoint] = "KeyPoint";
		map[TypeLine] = "Line";
		map[TypeQuadratic] = "Quadratic";
		map[TypeCubic] = "Cubic";
		map[TypeGroup] = "Group";
		map[TypePanelPolygon] = "PanelPolygon";
		map[TypeSewing] = "Sewing";
		return map;
	}
	std::hash_map<AbstractPanelObject::Type, std::string> AbstractPanelObject::s_typeStringMap
		 = AbstractPanelObject::generateTypeStringMap();

	AbstractShape* AbstractShape::create(Type type, size_t id)
	{
		switch (type)
		{
		case ldp::AbstractShape::TypeLine:
			return new Line(id);
		case ldp::AbstractShape::TypeQuadratic:
			return new Quadratic(id);
		case ldp::AbstractShape::TypeCubic:
			return new Cubic(id);
		default:
			return nullptr;
		}
	}

	AbstractShape* AbstractShape::create(std::string typeString, size_t id)
	{
		for (auto& iter : s_typeStringMap)
		{
			if (iter.second == typeString)
				return create(iter.first, id);
		}
		return nullptr;
	}

	static std::pair<int, float> lineFitting(const std::vector<Float2>& pts, Line& line)
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

		KeyPoint kp = line.getKeyPoint(0);
		kp.position = (pts[0] - ct).dot(dir) * dir + ct;
		line.setKeyPoint(0, kp);
		kp = line.getKeyPoint(1);
		kp.position = (pts.back() - ct).dot(dir) * dir + ct;
		line.setKeyPoint(1, kp);

		return std::make_pair(pos, err);
	}

	// bezier params
	inline float B0(float u)
	{
		return (1 - u)*(1 - u)*(1 - u);
	}

	inline float B1(float u)
	{
		return 3 * u * (1-u) * (1 - u);
	}

	inline float B2(float u)
	{
		return 3 * u * u * (1 - u);
	}

	inline float B3(float u)
	{
		return u*u*u;
	}

	static std::pair<int, float> quadraticFitting(const std::vector<Float2>& pts, Quadratic& curve)
	{
		const Float2 lTan = (pts[1] - pts[0]).normalize();
		const Float2 rTan = (pts[pts.size() - 2] - pts[pts.size() - 1]).normalize();
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i - 1];
		for (auto& l : paramLen)
			l /= paramLen.back();

		ldp::Float2 bezCurve[3] = {pts[0], 0, pts.back()};
		float area = rTan.cross(lTan);
		if (area == 0.f)
			bezCurve[1] = (pts[0] + pts.back()) * 0.5f;
		else
		{
			float tr = Float2(pts[0] - pts.back()).cross(lTan) / area;
			bezCurve[1] = pts.back() + tr * rTan;
		}

		for (int i = 0; i < curve.numKeyPoints(); i++)
		{
			KeyPoint kp = curve.getKeyPoint(i);
			kp.position = bezCurve[i];
			curve.setKeyPoint(i, kp);
		}

		// calculate distance
		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			auto p0 = curve.getPointByParam(paramLen[i]);
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

	static std::pair<int, float> cubicFitting(const std::vector<Float2>& pts, Cubic& curve)
	{
		const Float2 lTan = (pts[1] - pts[0]).normalize();
		const Float2 rTan = (pts[pts.size()-2] - pts[pts.size()-1]).normalize();
		std::vector<float> paramLen(pts.size(), 0);
		for (size_t i = 1; i < pts.size(); i++)
			paramLen[i] = (pts[i] - pts[i - 1]).length() + paramLen[i-1];
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
			float dist = (pts[0]-pts.back()).length() / 3.f;

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

		for (int i = 0; i < curve.numKeyPoints(); i++)
		{
			KeyPoint kp = curve.getKeyPoint(i);
			kp.position = bezCurve[i];
			curve.setKeyPoint(i, kp);
		}

		// calculate distance
		int pos = -1;
		float err = -1;
		for (size_t i = 0; i < pts.size(); i++)
		{
			auto p0 = curve.getPointByParam(paramLen[i]);
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

	void AbstractShape::create(std::vector<std::shared_ptr<AbstractShape>>& curves, 
		const std::vector<Float2>& keyPoints, float thre)
	{
		assert(keyPoints.size() >= 2);

		if (keyPoints.size() == 2)
		{
			std::vector<KeyPoint> pts;
			for (const auto& p : keyPoints)
			{
				KeyPoint kp;
				kp.position = p;
				pts.push_back(kp);
			}
			return curves.push_back(ShapePtr(new Line(pts)));
		}

		// 1. fitting line
		Line line;
		auto err = lineFitting(keyPoints, line);
		if (err.second < thre)
			return curves.push_back(ShapePtr(new Line(line)));

		// 2. quadratic fitting
		Quadratic quad;
		err = quadraticFitting(keyPoints, quad);
		if (err.second < thre)
			return curves.push_back(ShapePtr(new Quadratic(quad)));

		// 3. cubic fitting
		Cubic cub;
		err = cubicFitting(keyPoints, cub);
		if (err.second < thre)
			return curves.push_back(ShapePtr(new Cubic(cub)));

		// 4. recursive fitting
		std::vector<Float2> left(keyPoints.begin(), keyPoints.begin() + err.first + 1);
		std::vector<Float2> right(keyPoints.begin() + err.first, keyPoints.end());
		create(curves, left, thre);
		create(curves, right, thre);
	}

	AbstractShape* AbstractShape::clone()const
	{
		auto shape = create(getType(), getId());
		shape->m_keyPoints = m_keyPoints;
		for (auto& p : shape->m_keyPoints)
			p.reset((KeyPoint*)p->clone());
		shape->m_selected = m_selected;
		shape->m_highlighted = false;
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
		m_length = getLength();
		m_invalid = false;
		return m_samplePoints;
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

	PanelPolygon::PanelPolygon(size_t id) : AbstractPanelObject(id)
	{
		m_bbox[0] = FLT_MAX;
		m_bbox[1] = -FLT_MAX;
	}

	PanelPolygon::~PanelPolygon()
	{
	
	}

	void PanelPolygon::clear()
	{
		m_outerPoly.reset((Polygon*)nullptr);
		m_darts.clear();
		m_innerLines.clear();
	}

	void PanelPolygon::create(PolygonPtr outerPoly)
	{
		m_outerPoly = outerPoly;
	}

	void PanelPolygon::addDart(DartPtr dart)
	{
		m_darts.push_back(dart);
	}

	void PanelPolygon::addInnerLine(InnerLinePtr line)
	{
		m_innerLines.push_back(line);
	}

	void PanelPolygon::updateBound(Float2& bmin, Float2& bmax)
	{
		m_bbox[0] = FLT_MAX;
		m_bbox[1] = -FLT_MAX;
		if (m_outerPoly)
			m_outerPoly->updateBound(m_bbox[0], m_bbox[1]);
		for (auto& dart : m_darts)
			dart->updateBound(m_bbox[0], m_bbox[1]);
		for (auto& ln : m_innerLines)
			ln->updateBound(m_bbox[0], m_bbox[1]);
		for (int k = 0; k < bmin.size(); k++)
		{
			bmin[k] = std::min(bmin[k], m_bbox[0][k]);
			bmax[k] = std::max(bmax[k], m_bbox[1][k]);
		}
	}

	bool PanelPolygon::select(int idx, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		std::set<int> idxSet;
		idxSet.insert(idx);
		return select(idxSet, op);
	}

	bool PanelPolygon::select(const std::set<int>& idxSet, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		m_tmpbufferObj.clear();
		collectObject(m_tmpbufferObj);
		bool changed = false;
		for (auto& obj : m_tmpbufferObj)
		{
			bool oldSel = obj->isSelected();
			switch (op)
			{
			case ldp::AbstractPanelObject::SelectThis:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				else
					obj->setSelected(false);
				break;
			case ldp::AbstractPanelObject::SelectUnion:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				break;
			case ldp::AbstractPanelObject::SelectUnionInverse:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(!obj->isSelected());
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
			bool newSel = obj->isSelected();
			if (oldSel != newSel)
				changed = true;
		} // end for obj
		return changed;
	}

	void PanelPolygon::highLight(int idx, int lastIdx)
	{
		auto cur = getPtrById(idx);
		if (cur)
			cur->setHighlighted(true);
		if (idx != lastIdx)
		{
			auto pre = getPtrById(lastIdx);
			if (pre)
				pre->setHighlighted(false);
		}
	}

	void PanelPolygon::collectObject(std::vector<AbstractPanelObject*>& objs)
	{
		objs.push_back(this);
		if (m_outerPoly)
			m_outerPoly->collectObject(objs);
		for (auto& dart : m_darts)
			dart->collectObject(objs);
		for (auto& sp : m_innerLines)
			sp->collectObject(objs);
	}

	void PanelPolygon::collectObject(std::vector<const AbstractPanelObject*>& objs)const
	{
		objs.push_back(this);
		m_outerPoly->collectObject(objs);
		for (auto& dart : m_darts)
			dart->collectObject(objs);
		for (auto& sp : m_innerLines)
			sp->collectObject(objs);
	}

	AbstractPanelObject* PanelPolygon::clone()const
	{
		PanelPolygon* obj = new PanelPolygon(*this);
		obj->m_tmpbufferObj.clear();
		if (obj->m_outerPoly)
			obj->m_outerPoly.reset((Polygon*)obj->m_outerPoly->clone());
		for (auto& p : obj->m_darts)
			p.reset((Dart*)p->clone());
		for (auto& p : obj->m_innerLines)
			p.reset((InnerLine*)p->clone());
		return obj;
	}

	TiXmlElement* PanelPolygon::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractPanelObject::toXML(parent);
		if (m_outerPoly)
		{
			TiXmlElement* poly = new TiXmlElement("OuterPoly");
			ele->LinkEndChild(poly);
			m_outerPoly->toXML(poly);
		}
		if (!m_darts.empty())
		{
			TiXmlElement* poly = new TiXmlElement("Darts");
			ele->LinkEndChild(poly);
			for (auto& dart : m_darts)
				dart->toXML(poly);
		}
		if (!m_innerLines.empty())
		{
			TiXmlElement* poly = new TiXmlElement("InnerLines");
			ele->LinkEndChild(poly);
			for (auto& line : m_innerLines)
				line->toXML(poly);
		}
		return ele;
	}
	
	void PanelPolygon::fromXML(TiXmlElement* self)
	{
		AbstractPanelObject::fromXML(self);
		clear();
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			if (child->Value() == std::string("OuterPoly"))
			{
				auto child1 = child->FirstChildElement();
				if (child1)
				{
					m_outerPoly.reset(new Polygon);
					m_outerPoly->fromXML(child1);
				}
			}
			if (child->Value() == std::string("Darts"))
			{
				ShapeGroup tmpShape;
				for (auto childchild = child->FirstChildElement(); childchild;
					childchild = childchild->NextSiblingElement())
				{
					if (childchild->Value() == tmpShape.getTypeString())
					{
						m_darts.push_back(std::shared_ptr<ShapeGroup>(new ShapeGroup));
						m_darts.back()->fromXML(childchild);
					}
				}
			}
			if (child->Value() == std::string("InnerLines"))
			{
				ShapeGroup tmpShape;
				for (auto childchild = child->FirstChildElement(); childchild;
					childchild = childchild->NextSiblingElement())
				{
					if (childchild->Value() == tmpShape.getTypeString())
					{
						m_innerLines.push_back(std::shared_ptr<ShapeGroup>(new ShapeGroup));
						m_innerLines.back()->fromXML(childchild);
					}
				}
			}
		}
		updateBound(m_bbox[0], m_bbox[1]);
	}

	void PanelPolygon::translate(ldp::Float2 t)
	{
		if (m_outerPoly)
			m_outerPoly->translate(t);
		for (auto& dart : m_darts)
			dart->translate(t);
		for (auto& sp : m_innerLines)
			sp->translate(t);
	}
	void PanelPolygon::rotate(Mat2f R)
	{
		if (m_outerPoly)
			m_outerPoly->rotate(R);
		for (auto& dart : m_darts)
			dart->rotate(R);
		for (auto& sp : m_innerLines)
			sp->rotate(R);
	}
	void PanelPolygon::rotateBy(Mat2f R, Float2 center)
	{
		if (m_outerPoly)
			m_outerPoly->rotateBy(R, center);
		for (auto& dart : m_darts)
			dart->rotateBy(R, center);
		for (auto& sp : m_innerLines)
			sp->rotateBy(R, center);
	}
	void PanelPolygon::scaleBy(Float2 S, Float2 center)
	{
		if (m_outerPoly)
			m_outerPoly->scaleBy(S, center);
		for (auto& dart : m_darts)
			dart->scaleBy(S, center);
		for (auto& sp : m_innerLines)
			sp->scaleBy(S, center);
	}
	void PanelPolygon::scale(Float2 S)
	{
		if (m_outerPoly)
			m_outerPoly->scale(S);
		for (auto& dart : m_darts)
			dart->scale(S);
		for (auto& sp : m_innerLines)
			sp->scale(S);
	}
	void PanelPolygon::transform(Mat3f M)
	{
		if (m_outerPoly)
			m_outerPoly->transform(M);
		for (auto& dart : m_darts)
			dart->transform(M);
		for (auto& sp : m_innerLines)
			sp->transform(M);
	}

	//////////////////////////////////////////////////////////////////////////////////
	void Sewing::clear()
	{
		m_firsts.clear();
		m_seconds.clear();
	}

	void Sewing::addFirst(Unit unit)
	{
		auto iter_same = m_firsts.end();
		for (auto iter = m_firsts.begin(); iter != m_firsts.end(); ++iter)
		{
			if (iter->id == unit.id)
			{
				iter_same = iter;
				break;
			}
		}
		if (iter_same == m_firsts.end())
			m_firsts.push_back(unit);
		else
			iter_same->reverse = unit.reverse;
	}

	void Sewing::addSecond(Unit unit)
	{
		auto iter_same = m_seconds.end();
		for (auto iter = m_seconds.begin(); iter != m_seconds.end(); ++iter)
		{
			if (iter->id == unit.id)
			{
				iter_same = iter;
				break;
			}
		}
		if (iter_same == m_seconds.end())
			m_seconds.push_back(unit);
		else
			iter_same->reverse = unit.reverse;
	}

	void Sewing::addFirsts(const std::vector<Unit>& unit)
	{
		for (const auto& u : unit)
			addFirst(u);
	}

	void Sewing::addSeconds(const std::vector<Unit>& unit)
	{
		for (const auto& u : unit)
			addSecond(u);
	}

	bool Sewing::isSame_ignoreOrder(const Sewing& rhs)const
	{
		if (m_firsts.size() != rhs.m_firsts.size()
			|| m_seconds.size() != rhs.m_seconds.size())
			return false;
		std::set<size_t> fset, sset;
		for (const auto& f : m_firsts)
			fset.insert(f.id);
		for (const auto& s : m_seconds)
			sset.insert(s.id);

		for (const auto& f : rhs.m_firsts)
		if (fset.find(f.id) == fset.end())
			return false;
		for (const auto& s : rhs.m_seconds)
		if (sset.find(s.id) == sset.end())
			return false;

		return true;
	}

	void Sewing::remove(size_t id)
	{
		std::set<size_t> shapes;
		shapes.insert(id);
		remove(shapes);
	}

	void Sewing::remove(const std::set<size_t>& s)
	{
		auto tmp = m_firsts;
		m_firsts.clear();
		for (auto& f : tmp)
		{
			if (s.find(f.id) == s.end())
				m_firsts.push_back(f);
		}
		tmp = m_seconds;
		m_seconds.clear();
		for (auto& f : tmp)
		{
			if (s.find(f.id) == s.end())
				m_seconds.push_back(f);
		}
	}

	bool Sewing::select(int idx, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		std::set<int> idxSet;
		idxSet.insert(idx);
		return select(idxSet, op);
	}

	bool Sewing::select(const std::set<int>& idxSet, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		m_tmpbufferObj.clear();
		collectObject(m_tmpbufferObj);
		bool changed = false;
		for (auto& obj : m_tmpbufferObj)
		{
			bool oldSel = obj->isSelected();
			switch (op)
			{
			case ldp::AbstractPanelObject::SelectThis:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				else
					obj->setSelected(false);
				break;
			case ldp::AbstractPanelObject::SelectUnion:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				break;
			case ldp::AbstractPanelObject::SelectUnionInverse:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(!obj->isSelected());
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
			bool newSel = obj->isSelected();
			if (oldSel != newSel)
				changed = true;
		} // end for obj
		return changed;
	}

	void Sewing::highLight(int idx, int lastIdx)
	{
		auto cur = getPtrById(idx);
		if (cur)
			cur->setHighlighted(true);
		if (idx != lastIdx)
		{
			auto pre = getPtrById(lastIdx);
			if (pre)
				pre->setHighlighted(false);
		}
	}

	Sewing* Sewing::clone() const
	{ 
		auto r =  new Sewing(*this); 
		r->m_tmpbufferObj.clear();
		return r;
	}

	TiXmlElement* Sewing::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractPanelObject::toXML(parent);
		TiXmlElement* fele = new TiXmlElement("Firsts");
		ele->LinkEndChild(fele);
		for (const auto& f : m_firsts)
		{
			TiXmlElement* uele = new TiXmlElement("unit");
			fele->LinkEndChild(uele);
			uele->SetAttribute("shape_id", f.id);
			uele->SetAttribute("reverse", f.reverse);
		}
		TiXmlElement* sele = new TiXmlElement("Seconds");
		ele->LinkEndChild(sele);
		for (const auto& s : m_seconds)
		{
			TiXmlElement* uele = new TiXmlElement("unit");
			sele->LinkEndChild(uele);
			uele->SetAttribute("shape_id", s.id);
			uele->SetAttribute("reverse", s.reverse);
		}
		return ele;
	}

	void Sewing::fromXML(TiXmlElement* self)
	{
		AbstractPanelObject::fromXML(self);
		clear();
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			if (child->Value() == std::string("Firsts"))
			{
				for (auto child1 = child->FirstChildElement(); child1; child1 = child1->NextSiblingElement())
				{
					if (child1->Value() == std::string("unit"))
					{
						Unit u;
						int tmp = 0;
						if (!child1->Attribute("shape_id", &tmp))
							throw std::exception("unit id lost");
						u.id = tmp;
						if (!child1->Attribute("reverse", &tmp))
							throw std::exception("unit reverse lost");
						u.reverse = !!tmp;
						m_firsts.push_back(u);
					}
				}
			}
			if (child->Value() == std::string("Seconds"))
			{
				for (auto child1 = child->FirstChildElement(); child1; child1 = child1->NextSiblingElement())
				{
					if (child1->Value() == std::string("unit"))
					{
						Unit u;
						int tmp = 0;
						if (!child1->Attribute("shape_id", &tmp))
							throw std::exception("unit id lost");
						u.id = tmp;
						if (!child1->Attribute("reverse", &tmp))
							throw std::exception("unit reverse lost");
						u.reverse = !!tmp;
						m_seconds.push_back(u);
					}
				}
			}
		}
	}
}