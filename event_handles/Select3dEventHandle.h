#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "Abstract3dEventHandle.h"
class Select3dEventHandle : public Abstract3dEventHandle
{
public:
	Select3dEventHandle(Viewer3d* v);
	~Select3dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeSelect; }
	virtual void mousePressEvent(QMouseEvent *);
	virtual void mouseReleaseEvent(QMouseEvent *);
	virtual void mouseDoubleClickEvent(QMouseEvent *);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent*);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);
	virtual void handleEnter();
	virtual void handleLeave();
};