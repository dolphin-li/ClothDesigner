#include "GraphLoop.h"
#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{
	GraphLoop::GraphLoop() : AbstractGraphObject()
	{
	
	}

	GraphLoop* GraphLoop::clone()const
	{
		GraphLoop* loop = (GraphLoop*)create(getType());
		loop->m_startEdge = m_startEdge;
		loop->setSelected(isSelected());
		loop->m_isBoundingLoop = m_isBoundingLoop;
		return loop;
	}

	bool GraphLoop::isClosed()const
	{
		auto edge = m_startEdge;
		do
		{
			edge = edge->nextEdge();
		} while (edge && edge != m_startEdge);
		return edge;
	}

	void GraphLoop::samplePoints(std::vector<Float2>& pts, float step, float angleThreCos)
	{
		pts.clear();
		auto edge = m_startEdge;
		do
		{
			const std::vector<Float2>& samples = edge->samplePointsOnShape(step);
			if (pts.size())
			{
				for (size_t i = 0; i < samples.size(); i++)
				{
					Float2 p = samples[i];
					// if too close
					if ((p - pts.back()).length() < step || (p - pts[0]).length() < step)
						continue;
					pts.push_back(p);
				} // end for i
			}
			edge = edge->nextEdge();
		} while (edge && edge != m_startEdge);
	}

	TiXmlElement* GraphLoop::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);

		ele->SetAttribute("bounding-loop", m_isBoundingLoop);

		auto edge = m_startEdge;
		do
		{
			TiXmlElement* child = new TiXmlElement(edge->getTypeString().c_str());
			ele->LinkEndChild(child);
			child->SetAttribute("id", edge->getId());
			edge = edge->nextEdge();
		} while (edge && edge != m_startEdge);
		return ele;
	}

	void GraphLoop::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);

		int bd = 0;
		if (self->Attribute("bounding-loop", &bd))
		{
			m_isBoundingLoop = !!bd;
		}

		// collect curves
		std::vector<AbstractGraphCurve*> curves;
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			int id = 0;
			if (!child->Attribute("id", &id))
				throw std::exception(("cannot find id for " + getTypeString()).c_str());
			auto obj = getObjByIdx_loading(id);
			assert(obj);
			if (!obj->isCurve())
				throw std::exception("type error");
			curves.push_back((AbstractGraphCurve*)obj);
		} // end for child
		assert(curves.size());

		// connect to next edge
		m_startEdge = curves[0];
		for (size_t i = 0; i+1 < curves.size(); i++)
			curves[i]->nextEdge() = curves[i+1];
		if (curves.back()->getEndPoint() == curves[0]->getStartPoint())
			curves.back()->nextEdge() = curves[0];

		// connect to verts
		for (AbstractGraphCurve* c : curves)
		{
			for (int i = 0; i < c->numKeyPoints(); i++)
				c->keyPoint(i)->edges().insert(c);
		}
	}

}