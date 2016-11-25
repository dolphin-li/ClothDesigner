#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "Abstract3dEventHandle.h"
class Translate3dEventHandle : public Abstract3dEventHandle
{
public:
	Translate3dEventHandle(Viewer3d* v);
	~Translate3dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeTranslate; }
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
};