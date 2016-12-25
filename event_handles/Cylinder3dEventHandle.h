#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "Abstract3dEventHandle.h"
class Cylinder3dEventHandle : public Abstract3dEventHandle
{
public:
	Cylinder3dEventHandle(Viewer3d* v);
	~Cylinder3dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeCylinder; }
	virtual void mousePressEvent(QMouseEvent *);
	virtual void mouseReleaseEvent(QMouseEvent *);
	virtual void mouseDoubleClickEvent(QMouseEvent *);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent*);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);
	virtual void handleEnter();
	virtual void handleLeave();
private:
	float m_axisScale;
	ldp::Float3 m_cylinderAxisMouseClick;
	float m_cylinderRadiusMouseClick = 0;
	bool m_transformed = false;
};