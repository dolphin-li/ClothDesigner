#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "ldpMat\ldp_basic_mat.h"
class Viewer2d;
class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
class ObjMesh;
class Abstract2dEventHandle
{
public:
	struct PickInfo
	{
	};
	enum ProcessorType{
		ProcessorTypeGeneral = 0,
		ProcessorTypeEditPattern,
		ProcessorTypeEnd, // the end, no processor for this
	};
public:
	Abstract2dEventHandle(Viewer2d* v);
	~Abstract2dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeGeneral; }
	static Abstract2dEventHandle* create(ProcessorType type, Viewer2d* v);
	QCursor& cursor(){ return m_cursor; }
	const QCursor& cursor()const{ return m_cursor; }
	QString iconFile()const;
	QString inactiveIconFile()const;
	QString toolTips()const;

	void pick(QPoint pos);
	const PickInfo& pickInfo()const { return m_pickInfo; }

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
	Viewer2d* m_viewer;
	int m_lastHighlightShapeId;
	int m_currentSelectedId;
	QPoint m_mouse_press_pt;
	QCursor m_cursor;
	QString m_iconFile;
	QString m_inactiveIconFile;
	QString m_toolTips;

	PickInfo m_pickInfo;
};