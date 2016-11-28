#pragma once
#include "ldpMat\ldp_basic_mat.h"
namespace ldp
{
	class AbstractShape
	{
	public:
		typedef ldp::Float2 Point;
		enum Type
		{
			TypeLine,
			TypeCubic,
			TypeGeneralCurve,
		};
	public:
		virtual Type getType()const = 0;
		virtual Point startPoint()const = 0;
		virtual Point endPoint()const = 0;
	protected:
	};

	class Line : public AbstractShape
	{
		
	};

	class Cubic : public AbstractShape
	{

	};

	class GeneralCurve : public AbstractShape
	{

	};

	class PanelPolygon
	{
	public:
		typedef AbstractShape::Point Point;
		struct InnerDart
		{
			std::vector<std::shared_ptr<AbstractShape>> data;
		};
	public:
		PanelPolygon();
		~PanelPolygon();

	private:
		std::vector<std::shared_ptr<AbstractShape>> m_outerPoly;		// p0p1p2...pnp0
		std::vector<InnerDart> m_innerDarts;
		std::vector<std::shared_ptr<AbstractShape>> m_innerLines;
	};
}