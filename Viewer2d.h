#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"
#include "event_handles\Abstract2dEventHandle.h"
#include "cloth\clothManager.h"
#include "cloth\graph\GraphsSewing.h"
#include <stack>
class ClothDesigner;
namespace ldp
{
	class ClothPiece;
	class AbstractGraphCurve;
}
enum UiSewAddingState
{
	UiSewAddingFirst,
	UiSewAddingSecond,
	UiSewAddingEnd
};
enum UiSewChanges
{
	UiSewNoChange,
	UiSewAddedToPanel,
	UiSewUiTmpChanged,
};
struct UiSewData
{
	std::vector<ldp::GraphsSewing::Unit> firsts;
	std::vector<ldp::GraphsSewing::Unit> seconds;
	ldp::GraphsSewing::Unit f, s;
	UiSewAddingState state = UiSewAddingEnd;
};
class Viewer2d : public QGLWidget
{
	Q_OBJECT

public:
public:
	Viewer2d(QWidget *parent);
	~Viewer2d();

	void init(ldp::ClothManager* clothManager, ClothDesigner* ui);
	ldp::ClothManager* getManager() { return m_clothManager; }
	ClothDesigner* getMainUI() { return m_mainUI; }
	const ldp::Camera& camera()const{ return m_camera; }
	ldp::Camera& camera(){ return m_camera; }
	void resetCamera();
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
	Qt::MouseButtons buttons()const{ return m_buttons; }
	QPoint lastMousePos()const{ return m_lastPos; }
	const QImage& fboImage()const{ return m_fboImage; }
	Abstract2dEventHandle::ProcessorType getEventHandleType()const;
	void setEventHandleType(Abstract2dEventHandle::ProcessorType type);
	const Abstract2dEventHandle* getEventHandle(Abstract2dEventHandle::ProcessorType type)const;
	Abstract2dEventHandle* getEventHandle(Abstract2dEventHandle::ProcessorType type);
	void beginDragBox(QPoint p);
	void endDragBox();
	bool isDragBoxMode()const { return m_isDragBox; }
	void beginSewingMode();
	void endSewingMode();
	bool isSewingMode()const { return m_isSewingMode; }
	UiSewChanges makeSewUnit(ldp::AbstractGraphCurve* curve, QPoint pos, bool tmp = false);
	UiSewAddingState getSewAddingState()const { return m_uiSews.state; }
	UiSewChanges setSewAddingState(UiSewAddingState s);
	UiSewChanges setNextSewAddingState();
	UiSewChanges deleteCurrentUISew();
	const UiSewData& getUiSewData()const { return m_uiSews; }
	void setUiSewData(const UiSewData& data) { m_uiSews = data; }

	int fboRenderedIndex(QPoint p)const;
	void getModelBound(ldp::Float3& bmin, ldp::Float3& bmax)const;
protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mouseDoubleClickEvent(QMouseEvent *ev);
	void wheelEvent(QWheelEvent*);
	void keyPressEvent(QKeyEvent*);
	void keyReleaseEvent(QKeyEvent*);
	void renderDragBox();
	void renderSelectionOnFbo();
	void renderBackground();
	void renderClothsPanels(bool idxMode);
	void renderClothsPanels_Edge(const ldp::ClothPiece* piece, bool idxMode);
	void renderClothsPanels_KeyPoint(const ldp::ClothPiece* piece, bool idxMode);
	void renderClothsSewing(bool idxMode);
	void renderMeshes(bool idxMode);
	void renderOneSew(const ldp::GraphsSewing* sew, bool idxMode);
	void renderOneSewUnit(const ldp::GraphsSewing::Unit& sew, bool idxMode);

	UiSewChanges addCurrentUISew();
protected:
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType;
	Qt::MouseButtons m_buttons;
	QGLFramebufferObject* m_fbo;
	QImage m_fboImage;
	bool m_isDragBox;
	QPoint m_dragBoxBegin;
	bool m_isSewingMode;
	Abstract2dEventHandle* m_currentEventHandle;
	std::vector<std::shared_ptr<Abstract2dEventHandle>> m_eventHandles;
	ldp::ClothManager* m_clothManager;
	ClothDesigner* m_mainUI;

	UiSewData m_uiSews;
};

