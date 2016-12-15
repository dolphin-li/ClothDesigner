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
class Transform2dPatternEventHandle : public Abstract2dEventHandle
{
public:
	Transform2dPatternEventHandle(Viewer2d* v);
	~Transform2dPatternEventHandle();
	virtual ProcessorType type()const { return ProcessorTypeTransformPattern; }
	virtual void mousePressEvent(QMouseEvent *);
	virtual void mouseReleaseEvent(QMouseEvent *);
	virtual void mouseDoubleClickEvent(QMouseEvent *);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent*);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);
	virtual void handleEnter();
	virtual void handleLeave();
protected:
	void panelLevelTransform_MouseMove(QMouseEvent* ev);
protected:
	bool m_transformed = false;
	ldp::Float2 m_translateStart;
	ldp::Float2 m_rotateCenter;
};