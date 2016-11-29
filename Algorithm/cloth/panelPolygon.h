#pragma once
#include "ldpMat\ldp_basic_mat.h"
#include <set>
namespace ldp
{
#pragma region --units
	class AbstractPanelObject
	{
	public:
		enum Type
		{
			TypeKeyPoint = 0x01,
			TypeLine = 0x02,
			TypeQuadratic = 0x04,
			TypeCubic = 0x08,
			TypeGeneralCurve = 0x10,
			TypeGroup = 0x20,
			TypePanelPolygon = 0x40,
		};
		enum SelectOp
		{
			SelectThis,
			SelectUnion,
			SelectUnionInverse,
			SelectAll,
			SelectNone,
			SelectInverse,
			SelectEnd
		};
	public:
		AbstractPanelObject()
		{
			m_idxStart = -1;
			m_highlighted = false;
			m_selected = false;
		}
		void setIdxBegin(int i) { m_idxStart = i; }
		int getIdxBegin()const { return m_idxStart; }
		virtual int getIdxEnd()const = 0;
		void setSelected(bool s) { m_selected = s; }
		bool isSelected()const { return m_selected; }
		void setHighlighted(bool s) { m_highlighted = s; }
		bool isHighlighted()const { return m_highlighted; }
		virtual Type getType()const = 0;
		virtual void collectObject(std::vector<AbstractPanelObject*>& objs)=0;
	protected:
		int m_idxStart;
		bool m_selected;
		bool m_highlighted;
	};

	class KeyPoint : public AbstractPanelObject
	{
	public:
		KeyPoint() : AbstractPanelObject(){}
		KeyPoint(const Float2& p) : AbstractPanelObject(), position(p){}
		virtual int getIdxEnd()const { return getIdxBegin() + 1; }
		KeyPoint* clone()const 
		{
			KeyPoint* p = new KeyPoint();
			p->m_highlighted = m_highlighted;
			p->m_idxStart = m_idxStart;
			p->m_selected = m_selected;
			p->position = position;
			return p;
		}
		Type getType()const { return TypeKeyPoint; }
		virtual void collectObject(std::vector<AbstractPanelObject*>& objs)
		{
			objs.push_back(this);
		}
		Float2 position;
	};

	class AbstractShape : public AbstractPanelObject
	{
	public:
		AbstractShape() : AbstractPanelObject()
		{
			m_invalid = true;
			m_lastSampleStep = 0;
		}
		static AbstractShape* create(Type type);
		static AbstractShape* create(const std::vector<Float2>& keyPoints);
		AbstractShape(const std::vector<KeyPoint>& keyPoints) : AbstractShape()
		{
			m_keyPoints.clear();
			for (const auto& p : keyPoints)
			{
				m_keyPoints.push_back(std::shared_ptr<KeyPoint>(new KeyPoint(p)));
			}
		}
		virtual AbstractShape* clone()const;
		virtual int getIdxEnd()const 
		{ 
			if (m_keyPoints.size() == 0)
				return m_idxStart + 1;
			return
				m_keyPoints.back()->getIdxEnd();
		}
		virtual Type getType()const = 0;
		virtual Float2 getPointByParam(float t)const = 0; // t \in [0, 1]
		virtual float calcLength()const;
		const KeyPoint& getStartPoint()const
		{
			return *m_keyPoints[0];
		}
		const KeyPoint& getEndPoint()const
		{
			return *m_keyPoints.back();
		}
		int numKeyPoints()
		{
			return (int)m_keyPoints.size();
		}
		void setKeyPoint(int i, KeyPoint p)
		{
			m_invalid = true;
			*m_keyPoints[i] = p;
		}
		const KeyPoint& getKeyPoint(int i)const
		{
			return *m_keyPoints[i];
		}
		virtual const std::vector<Float2>& samplePointsOnShape(float step)const;
		void translate(Float2 t)
		{
			for (auto& p : m_keyPoints)
				p->position += t;
		}
		void rotate(const ldp::Mat2f& R)
		{
			for (auto& p : m_keyPoints)
				p->position = R * p->position;
		}
		void rotateBy(const ldp::Mat2f& R, Float2 c)
		{
			for (auto& p : m_keyPoints)
				p->position = R * (p->position - c) + c;
		}
		void scale(Float2 s)
		{
			for (auto& p : m_keyPoints)
				p->position *= s;
		}
		void scaleBy(Float2 s, Float2 c)
		{
			for (auto& p : m_keyPoints)
				p->position = s*(p->position - c) + c;
		}
		void transform(const ldp::Mat3f& M)
		{
			for (auto& p : m_keyPoints)
			{
				ldp::Float3 p3(p->position[0], p->position[1], 1);
				p3 = M * p3;
				p->position[0] = p3[0] / p3[2];
				p->position[1] = p3[1] / p3[2];
			}
		}
		void unionBound(Float2& bmin, Float2& bmax)
		{
			for (const auto& p : m_keyPoints)
			{
				for (int k = 0; k < p->position.size(); k++)
				{
					bmin[k] = std::min(bmin[k], p->position[k]);
					bmax[k] = std::max(bmax[k], p->position[k]);
				}
			}
		}
		AbstractShape& reverse()
		{
			std::reverse(m_keyPoints.begin(), m_keyPoints.end());
			m_invalid = true;
			return *this;
		}
		void updateIndex(int idx)
		{
			m_idxStart = idx++; // self
			for (size_t i = 0; i < m_keyPoints.size(); i++)
			{
				m_keyPoints[i]->setIdxBegin(idx);
				idx = m_keyPoints[i]->getIdxEnd();
			}
		}
		virtual void collectObject(std::vector<AbstractPanelObject*>& objs)
		{
			objs.push_back(this);
			for (auto p : m_keyPoints)
				p->collectObject(objs);
		}
	protected:
		std::vector<std::shared_ptr<KeyPoint>> m_keyPoints;
		mutable std::vector<Float2> m_samplePoints;
		mutable bool m_invalid;
		mutable float m_lastSampleStep;
	};

	class Line : public AbstractShape
	{
	public:
		Line() : AbstractShape()
		{
			m_keyPoints.resize(2);
		}
		Line(const std::vector<KeyPoint>& keyPoints) : AbstractShape(keyPoints)
		{
			assert(keyPoints.size() == 2);
		}
		virtual Type getType()const { return TypeLine; }
		virtual Float2 getPointByParam(float t)const
		{
			return m_keyPoints[0]->position * (1 - t) + m_keyPoints[1]->position * t;
		}
		virtual float calcLength()const { return (m_keyPoints[1]->position - m_keyPoints[0]->position).length(); }
	};

	class Quadratic : public AbstractShape
	{
	public:
		Quadratic() : AbstractShape()
		{
			m_keyPoints.resize(3);
		}
		Quadratic(const std::vector<KeyPoint>& keyPoints) : AbstractShape(keyPoints)
		{
			assert(keyPoints.size() == 3);
		}
		virtual Type getType()const { return TypeQuadratic; }
		virtual Float2 getPointByParam(float t)const
		{
			return (1 - t) * ((1 - t)*m_keyPoints[0]->position + t*m_keyPoints[1]->position)
				+ t * ((1 - t)*m_keyPoints[1]->position + t*m_keyPoints[2]->position);
		}
	};

	class Cubic : public AbstractShape
	{
	public:
		Cubic() : AbstractShape()
		{
			m_keyPoints.resize(4);
		}
		Cubic(const std::vector<KeyPoint>& keyPoints) : AbstractShape(keyPoints)
		{
			assert(keyPoints.size() == 4);
		}
		virtual Type getType()const { return TypeCubic; }
		virtual Float2 getPointByParam(float t)const
		{
			Float2 p1 = (1 - t) * ((1 - t)*m_keyPoints[0]->position + t*m_keyPoints[1]->position)
				+ t * ((1 - t)*m_keyPoints[1]->position + t*m_keyPoints[2]->position);
			Float2 p2 = (1 - t) * ((1 - t)*m_keyPoints[1]->position + t*m_keyPoints[2]->position)
				+ t * ((1 - t)*m_keyPoints[2]->position + t*m_keyPoints[3]->position);
			return (1 - t) * p1 + t * p2;
		}
	};

	class GeneralCurve : public AbstractShape
	{
	public:
		GeneralCurve() : AbstractShape()
		{
			
		}
		GeneralCurve(const std::vector<KeyPoint>& keyPoints);
		virtual AbstractShape* clone()const;
		virtual Type getType()const { return TypeGeneralCurve; }
		virtual Float2 getPointByParam(float t)const;
		virtual float calcLength()const;
	protected:
		std::vector<float> m_params;
	};

	typedef std::shared_ptr<KeyPoint> KeyPointPtr;
	typedef std::shared_ptr<AbstractShape> ShapePtr;
	typedef std::shared_ptr<Line> LinePtr;
	typedef std::shared_ptr<Quadratic> QaudraticPtr;
	typedef std::shared_ptr<Cubic> CubicPtr;
	typedef std::shared_ptr<GeneralCurve> GeneralCurvePtr;

	class ShapeGroup : public std::vector<ShapePtr>, public AbstractPanelObject
	{
	public:
		ShapeGroup() : AbstractPanelObject(), std::vector<ShapePtr>()
		{
			m_bbox[0] = FLT_MAX;
			m_bbox[1] = -FLT_MAX;
		}
		virtual Type getType()const { return TypeGroup; }
		void cloneTo(ShapeGroup& rhs)const
		{
			rhs.resize(size());
			for (size_t i = 0; i < rhs.size(); i++)
				rhs[i].reset((*this)[i]->clone());
		}
		void updateBound(Float2& bmin, Float2& bmax)
		{
			m_bbox[0] = FLT_MAX;
			m_bbox[1] = FLT_MIN;
			for (size_t i = 0; i < size(); i++)
				(*this)[i]->unionBound(m_bbox[0], m_bbox[1]);
			for (int k = 0; k < bmin.size(); k++)
			{
				bmin[k] = std::min(bmin[k], m_bbox[0][k]);
				bmax[k] = std::max(bmax[k], m_bbox[1][k]);
			}
		}
		const Float2* bound()const { return m_bbox; }
		ShapeGroup& reverse()
		{
			for (size_t i = 0; i < size(); i++)
				(*this)[i]->reverse();
			std::reverse(begin(), end());
			return *this;
		}
		void updateIndex(int idx) 
		{
			m_idxStart = idx;
			idx++;	//self
			for (size_t i = 0; i < size(); i++)
			{
				(*this)[i]->updateIndex(idx);
				idx = (*this)[i]->getIdxEnd();
			}
		}
		virtual int getIdxEnd()const 
		{ 
			if (size())
				return back()->getIdxEnd(); 
			return m_idxStart+1;
		}
		void collectKeyPoints(std::vector<Float2>& pts, float distThre = 0)
		{
			for (size_t i = 0; i < size(); i++)
			{
				const auto& sp = (*this)[i];
				for (int j = 0; j < sp->numKeyPoints(); j++)
				{
					if (pts.size())
					{
						if ((pts.back() - sp->getKeyPoint(j).position).length() < distThre)
							continue;
					}
					pts.push_back(sp->getKeyPoint(j).position);
				} // end for j
			} // end for i
		}
		void collectSamplePoints(std::vector<Float2>& pts, float step, float distThre = 0)
		{
			for (size_t i = 0; i < size(); i++)
			{
				const auto& sp = (*this)[i];
				auto ps = sp->samplePointsOnShape(step);
				for (int j = 0; j < ps.size(); j++)
				{
					if (pts.size())
					{
						if ((pts.back() - ps[j]).length() < distThre)
							continue;
					}
					pts.push_back(sp->getKeyPoint(j).position);
				} // end for j
			} // end for i
		}
		virtual void collectObject(std::vector<AbstractPanelObject*>& objs)
		{
			objs.push_back(this);
			for (auto p : (*this))
				p->collectObject(objs);
		}
	protected:
		Float2 m_bbox[2];
	};
#pragma endregion

	class PanelPolygon : public AbstractPanelObject
	{
	public:
		typedef ShapeGroup Polygon;
		typedef ShapeGroup Dart;
	public:
		PanelPolygon();
		~PanelPolygon();

		void clear();

		virtual Type getType()const { return TypePanelPolygon; }
		void create(const Polygon& outerPoly, int idxStart);
		void addDart(Dart& dart);
		void addInnerLine(std::shared_ptr<AbstractShape> line);
		const std::vector<Dart>& darts()const { return m_darts; }
		const std::vector<ShapePtr>& innerLines()const { return m_innerLines; }
		const Polygon& outerPoly()const { return m_outerPoly; }

		// update the index of all units, starting from the given
		void updateIndex(int idx);
		virtual int getIdxEnd()const;

		void select(int idx, SelectOp op);
		void select(const std::set<int>& indices, SelectOp op);
		void highLight(int idx);

		void updateBound(Float2& bmin, Float2& bmax);
		const Float2* bound()const { return m_bbox; }
	protected:
		virtual void collectObject(std::vector<AbstractPanelObject*>& objs);
	private:
		Polygon m_outerPoly;		// p0p1p2...pnp0
		std::vector<Dart> m_darts;
		std::vector<ShapePtr> m_innerLines;
		ldp::Float2 m_bbox[2];
		std::vector<AbstractPanelObject*> m_tmpbufferObj;
	};
}