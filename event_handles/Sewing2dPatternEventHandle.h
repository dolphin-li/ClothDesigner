#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "ldpMat\ldp_basic_mat.h"
#include "Abstract2dEventHandle.h"
class Viewer2d;
class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
class ObjMesh;
class Sewing2dPatternEventHandle : public Abstract2dEventHandle
{
public:
	Sewing2dPatternEventHandle(Viewer2d* v);
	~Sewing2dPatternEventHandle();
	virtual ProcessorType type()const { return ProcessorTypeSewingPattern; }
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