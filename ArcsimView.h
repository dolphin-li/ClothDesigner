#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"

namespace arcsim
{
	class ArcSimManager;
}
class CShaderManager;
class MeshRender;
class GPUBuffers;
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
	std::shared_ptr<CShaderManager> m_shaderManager;

	arcsim::ArcSimManager* m_arcsimManager = nullptr;
	////////////////////////////////////////////////////////////////////////////////////
	// shadow map related
protected:
	enum{
		MAX_LIGHTS_PASS = 4
	};
	int m_lightNum = 0;
	bool m_showShadow = false;
	bool m_shadowMapInitialized = false;
	std::vector<ldp::Float3> m_lightOriginalColors;
	std::vector<ldp::Float3> m_lightShadeColors;
	std::vector<ldp::Float3> m_lightDirections;
	std::shared_ptr<MeshRender> m_MeshRender;
	std::shared_ptr<GPUBuffers> m_GPUBuffers;
	ldp::Mat4f m_LightModelViewMatrix[MAX_LIGHTS_PASS];
	ldp::Mat4f m_LightProjectionMatrix[MAX_LIGHTS_PASS];
	ldp::Mat4f m_LightModelViewProjectionMatrix[MAX_LIGHTS_PASS];

	void initShadowMap();
	bool loadLight(QString filename);
	void renderWithShadowMap();
};

