#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"
namespace ldp
{
	class ClothManager;
}
class ObjMesh;
class TrainingImageRenderView : public QGLWidget
{
	Q_OBJECT

public:
	TrainingImageRenderView(QWidget *parent);
	~TrainingImageRenderView();

	void init(ldp::ClothManager* clothManager, ObjMesh* clothMeshLoaded);
	const ldp::Camera& camera()const{ return m_camera; }
	ldp::Camera& camera(){ return m_camera; }
	void resetCamera();
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
	Qt::MouseButtons buttons()const{ return m_buttons; }
	QPoint lastMousePos()const{ return m_lastPos; }
	const QImage& fboImage()const{ return m_fboImage; }

	void setShowBody(bool s) { m_showBody = s; }
	bool isShowBody()const { return m_showBody; }

	void generateDistMap_x9(std::vector<QImage>& distMaps);
protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mouseDoubleClickEvent(QMouseEvent *ev);
	void wheelEvent(QWheelEvent*);
	void keyPressEvent(QKeyEvent*);
	void keyReleaseEvent(QKeyEvent*);
	void renderSelectionOnFbo();
protected:
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType;
	Qt::MouseButtons m_buttons;
	QGLFramebufferObject* m_fbo;
	QImage m_fboImage;
	bool m_showBody = true;

	ldp::ClothManager* m_clothManager = nullptr;
	ObjMesh* m_clothMeshLoaded = nullptr;
};

