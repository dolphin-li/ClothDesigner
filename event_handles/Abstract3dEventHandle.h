#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>

class Viewer3d;
class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
class Abstract3dEventHandle
{
public:
	enum ProcessorType{
		ProcessorTypeGeneral = 0,
		ProcessorTypeEnd, // the end, no processor for this
	};
public:
	Abstract3dEventHandle(Viewer3d* v);
	~Abstract3dEventHandle();
	virtual ProcessorType type() { return ProcessorTypeGeneral; }
	static Abstract3dEventHandle* create(ProcessorType type, Viewer3d* v);
	QCursor& cursor(){ return m_cursor; }
	const QCursor& cursor()const{ return m_cursor; }
	QString iconFile()const;
	QString inactiveIconFile()const;
	QString toolTips()const;

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
	Viewer3d* m_viewer;
	int m_lastHighlightShapeId;
	int m_currentSelectedId;
	QPoint m_mouse_press_pt;
	QCursor m_cursor;
	QString m_iconFile;
	QString m_inactiveIconFile;
	QString m_toolTips;
	ldp::Double3 m_picked_screenPos;
};