#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "Abstract3dEventHandle.h"
class Rotate3dEventHandle : public Abstract3dEventHandle
{
public:
	Rotate3dEventHandle(Viewer3d* v);
	~Rotate3dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeRotate; }
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
	ldp::Mat3f m_trackBallMouseClickR;
};