#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"
#include "event_handles\Abstract2dEventHandle.h"
#include "cloth\clothManager.h"
class Viewer2d : public QGLWidget
{
	Q_OBJECT

public:
	Viewer2d(QWidget *parent);
	~Viewer2d();

	void init(ldp::ClothManager* clothManager);
	ldp::ClothManager* getManager() { return m_clothManager; }
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
protected:
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType;
	Qt::MouseButtons m_buttons;
	QGLFramebufferObject* m_fbo;
	QImage m_fboImage;
	bool m_isDragBox;
	QPoint m_dragBoxBegin;
	Abstract2dEventHandle* m_currentEventHandle;
	std::vector<std::shared_ptr<Abstract2dEventHandle>> m_eventHandles;
	ldp::ClothManager* m_clothManager;
};

