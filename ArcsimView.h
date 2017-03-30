#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"

namespace arcsim
{
	class ArcSimManager;
}
class ArcsimView : public QGLWidget
{
	Q_OBJECT

public:
	ArcsimView(QWidget *parent);
	~ArcsimView();

	void init(arcsim::ArcSimManager* manager);

	const ldp::Camera& camera()const{ return m_camera; }
	ldp::Camera& camera(){ return m_camera; }
	void resetCamera();
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

	void setShowBody(bool s) { m_showBody = s; }
	bool isShowBody()const { return m_showBody; }

protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mouseDoubleClickEvent(QMouseEvent *ev);
	void wheelEvent(QWheelEvent*);
	void keyPressEvent(QKeyEvent*);
	void keyReleaseEvent(QKeyEvent*);
protected:
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType;
	bool m_showBody = true;

	arcsim::ArcSimManager* m_arcsimManager = nullptr;
};

