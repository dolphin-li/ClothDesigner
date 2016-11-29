#pragma once
#include "ldpMat\ldp_basic_mat.h"
namespace ldp
{
#pragma region --units
	class AbstractShape
	{
	public:
		typedef ldp::Float2 Point;
		enum Type
		{
			TypeLine,
			TypeQuadratic,
			TypeCubic,
			TypeGeneralCurve,
		};
	public:
		AbstractShape()
		{
			m_idxStart = -1;
			m_highlighted = false;
			m_selected = false;
			m_invalid = true;
			m_lastSampleStep = 0;
		}
		static AbstractShape* create(Type type);
		static AbstractShape* create(const std::vector<Point>& keyPoints);
		AbstractShape* clone()const
		{
			auto shape = create(m_keyPoints);
			shape->m_idxStart = m_idxStart;
			shape->m_selected = m_selected;
			shape->m_highlighted = m_highlighted;
			return shape;
		}
		void setIdxBegin(int i) { m_idxStart = i; }
		int getIdxBegin()const { return m_idxStart; }
		int getIdxEnd()const { return m_idxStart + numIdx(); }
		int numIdx()const { return m_keyPoints.size() + 1; }
		void setSelected(bool s) { m_selected = s; }
		bool isSelected()const { return m_selected; }
		void setHighlighted(bool s) { m_highlighted = s; }
		bool isHighlighted()const { return m_highlighted; }
		virtual Type getType()const = 0;
		virtual Point getPointByParam(float t)const=0; // t \in [0, 1]
		virtual float calcLength()const;
		Point getStartPoint()const
		{
			return m_keyPoints[0];
		}
		Point getEndPoint()const
		{
			return m_keyPoints.back();
		}
		int numKeyPoints()
		{
			return (int)m_keyPoints.size();
		}
		void setKeyPoint(int i, Point p)
		{
			m_invalid = true;
			m_keyPoints[i] = p;
		}
		const Point& getKeyPoint(int i)const
		{
			return m_keyPoints[i];
		}
		virtual const std::vector<Point>& samplePointsOnShape(float step)const;
		void translate(Point t)
		{
			for (auto& p : m_keyPoints)
				p += t;
		}
		void rotate(const ldp::Mat2f& R)
		{
			for (auto& p : m_keyPoints)
				p = R * p;
		}
		void rotateBy(const ldp::Mat2f& R, Point c)
		{
			for (auto& p : m_keyPoints)
				p = R * (p - c) + c;
		}
		void scale(Point s)
		{
			for (auto& p : m_keyPoints)
				p *= s;
		}
		void scaleBy(Point s, Point c)
		{
			for (auto& p : m_keyPoints)
				p = s*(p - c) + c;
		}
		void transform(const ldp::Mat3f& M)
		{
			for (auto& p : m_keyPoints)
			{
				ldp::Float3 p3(p[0], p[1], 1);
				p3 = M * p3;
				p[0] = p3[0] / p3[2];
				p[1] = p3[1] / p3[2];
			}
		}
		void unionBound(Point& bmin, Point& bmax)
		{
			for (const auto& p : m_keyPoints)
			{
				for (int k = 0; k < p.size(); k++)
				{
					bmin[k] = std::min(bmin[k], p[k]);
					bmax[k] = std::max(bmax[k], p[k]);
				}
			}
		}
		AbstractShape& reverse()
		{
			std::reverse(m_keyPoints.begin(), m_keyPoints.end());
			m_invalid = true;
			return *this;
		}
	protected:
		std::vector<Point> m_keyPoints;
		int m_idxStart;
		bool m_selected;
		bool m_highlighted;
		mutable std::vector<Point> m_samplePoints;
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
		Line(const std::vector<Point>& keyPoints) : AbstractShape()
		{
			assert(keyPoints.size() == 2);
			m_keyPoints = keyPoints;
		}
		virtual Type getType()const { return TypeLine; }
		virtual Point getPointByParam(float t)const
		{
			return m_keyPoints[0] * (1 - t) + m_keyPoints[1] * t;
		}
		virtual float calcLength()const { return (m_keyPoints[1] - m_keyPoints[0]).length(); }
	};

	class Quadratic : public AbstractShape
	{
	public:
		Quadratic() : AbstractShape()
		{
			m_keyPoints.resize(3);
		}
		Quadratic(const std::vector<Point>& keyPoints) : AbstractShape()
		{
			assert(keyPoints.size() == 3);
			m_keyPoints = keyPoints;
		}
		virtual Type getType()const { return TypeQuadratic; }
		virtual Point getPointByParam(float t)const
		{
			return (1 - t) * ((1 - t)*m_keyPoints[0] + t*m_keyPoints[1])
				+ t * ((1 - t)*m_keyPoints[1] + t*m_keyPoints[2]);
		}
	};

	class Cubic : public AbstractShape
	{
	public:
		Cubic() : AbstractShape()
		{
			m_keyPoints.resize(4);
		}
		Cubic(const std::vector<Point>& keyPoints) : AbstractShape()
		{
			assert(keyPoints.size() == 4);
			m_keyPoints = keyPoints;
		}
		virtual Type getType()const { return TypeCubic; }
		virtual Point getPointByParam(float t)const
		{
			Point p1 = (1 - t) * ((1 - t)*m_keyPoints[0] + t*m_keyPoints[1])
				+ t * ((1 - t)*m_keyPoints[1] + t*m_keyPoints[2]);
			Point p2 = (1 - t) * ((1 - t)*m_keyPoints[1] + t*m_keyPoints[2])
				+ t * ((1 - t)*m_keyPoints[2] + t*m_keyPoints[3]);
			return (1 - t) * p1 + t * p2;
		}
	};

	class GeneralCurve : public AbstractShape
	{
	public:
		GeneralCurve() : AbstractShape()
		{
			
		}
		GeneralCurve(const std::vector<Point>& keyPoints);
		virtual Type getType()const { return TypeGeneralCurve; }
		virtual Point getPointByParam(float t)const;
		virtual float calcLength()const;
	protected:
		std::vector<float> m_params;
	};
	
	typedef std::shared_ptr<AbstractShape> ShapePtr;
	typedef std::shared_ptr<Line> LinePtr;
	typedef std::shared_ptr<Quadratic> QaudraticPtr;
	typedef std::shared_ptr<Cubic> CubicPtr;
	typedef std::shared_ptr<GeneralCurve> GeneralCurvePtr;

	class ShapeGroup : public std::vector<ShapePtr>
	{
		typedef AbstractShape::Point Point;
	public:
		void cloneTo(ShapeGroup& rhs)const
		{
			rhs.resize(size());
			for (size_t i = 0; i < rhs.size(); i++)
				rhs[i].reset((*this)[i]->clone());
		}
		void updateBound(Point& bmin, Point& bmax)
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
		const Point* bound()const { return m_bbox; }
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
				(*this)[i]->setIdxBegin(idx);
				idx = (*this)[i]->getIdxEnd();
			}
		}
		int getIdxBegin()const { return m_idxStart; }
		int getIdxEnd()const 
		{ 
			if (size())
				return back()->getIdxEnd(); 
			return m_idxStart+1;
		}
		void collectKeyPoints(std::vector<Point>& pts, float distThre = 0)
		{
			for (size_t i = 0; i < size(); i++)
			{
				const auto& sp = (*this)[i];
				for (int j = 0; j < sp->numKeyPoints(); j++)
				{
					if (pts.size())
					{
						if ((pts.back() - sp->getKeyPoint(j)).length() < distThre)
							continue;
					}
					pts.push_back(sp->getKeyPoint(j));
				} // end for j
			} // end for i
		}
		void collectSamplePoints(std::vector<Point>& pts, float step, float distThre = 0)
		{
			for (size_t i = 0; i < size(); i++)
			{
				const auto& sp = (*this)[i];
				auto ps = sp->samplePointsOnShape(step);
				for (int j = 0; j < ps.size(); j++)
				{
					if (pts.size())
					{
						if ((pts.back() - sp->getKeyPoint(j)).length() < distThre)
							continue;
					}
					pts.push_back(sp->getKeyPoint(j));
				} // end for j
			} // end for i
		}
	protected:
		Point m_bbox[2];
		int m_idxStart;
	};
#pragma endregion

	class PanelPolygon
	{
	public:
		typedef AbstractShape::Point Point;
		typedef ShapeGroup Polygon;
		typedef ShapeGroup Dart;
	public:
		PanelPolygon();
		~PanelPolygon();

		void clear();

		void create(const Polygon& outerPoly, int idxStart);
		void addDart(Dart& dart);
		void addInnerLine(std::shared_ptr<AbstractShape> line);
		const std::vector<Dart>& darts()const { return m_darts; }
		const std::vector<ShapePtr>& innerLines()const { return m_innerLines; }
		const Polygon& outerPoly()const { return m_outerPoly; }

		// update the index of all units, starting from the given
		void updateIndex(int idx);
		int getIndexBegin()const;
		int getIndexEnd()const;

		void updateBound(Point& bmin, Point& bmax);
		const Point* bound()const { return m_bbox; }
	private:
		Polygon m_outerPoly;		// p0p1p2...pnp0
		std::vector<Dart> m_darts;
		std::vector<ShapePtr> m_innerLines;
		int m_idxStart;
		ldp::Float2 m_bbox[2];
	};
}