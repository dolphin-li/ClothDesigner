#pragma once

#include <QPoint>
#include <QCursor>
#include <QString>
#include "ldpMat\ldp_basic_mat.h"
class Viewer3d;
class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
class ObjMesh;
namespace ldp
{
	class ClothPiece;
}
class Abstract3dEventHandle
{
public:
	struct PickInfo
	{
		int renderId = -1;
		ldp::Float3 screenPos;
		ObjMesh* mesh = nullptr;
		ldp::ClothPiece* piece = nullptr;
		int faceId = -1;
		ldp::Float3 pickInnerCoords;
		ldp::Float3 pickPos;
		ldp::Float3 meshCenter;
	};
	enum ProcessorType{
		ProcessorTypeGeneral = 0,
		ProcessorTypeSelect,
		ProcessorTypeTranslate,
		ProcessorTypeRotate,
		ProcessorTypeCylinder,
		ProcessorTypeSmpl,
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

	void resetSelection();

	void pick(QPoint pos);
	const PickInfo& pickInfo()const { return m_pickInfo; }
	void getSelectionRay(QPoint mousePos, ldp::Float3& p, ldp::Float3& q)const;

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

	PickInfo m_pickInfo;
};