#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"
#include "event_handles\Abstract3dEventHandle.h"
#include "cloth\clothManager.h"
#include <Shader\ShaderManager.h>
class ClothDesigner;
class Viewer3d : public QGLWidget
{
	Q_OBJECT

public:
	enum {
		FaceIndex = 1,
		TrackBallIndex_X = 0x3fffffff,
		TrackBallIndex_Y,
		TrackBallIndex_Z,
		SmplJointIndex = 0x41000002
	};
	enum TrackBallMode{
		TrackBall_None,
		TrackBall_Rot,
		TrackBall_Trans,
		TrackBall_Cylinder
	};
public:
	Viewer3d(QWidget *parent);
	~Viewer3d();

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
	Abstract3dEventHandle::ProcessorType getEventHandleType()const;
	void setEventHandleType(Abstract3dEventHandle::ProcessorType type);
	const Abstract3dEventHandle* getEventHandle(Abstract3dEventHandle::ProcessorType type)const;
	Abstract3dEventHandle* getEventHandle(Abstract3dEventHandle::ProcessorType type);
	void beginDragBox(QPoint p);
	void rotateTrackBall(ldp::Mat3d R);
	void translateTrackBall(ldp::Double3 t);
	void endDragBox();
	void beginTrackBall(TrackBallMode mode, ldp::Float3 p, ldp::Mat3f R, float scale);
	void endTrackBall();
	TrackBallMode getTrackBallMode()const{ return m_trackBallMode; }
	void setActiveTrackBallAxis(int i){ m_activeTrackBallAxis = i; }
	int getActiveTrackBallAxis()const{ return m_activeTrackBallAxis; }
	void setHoverTrackBallAxis(int i){ m_hoverTrackBallAxis = i; }
	int getHoverTrackBallAxis()const{ return m_hoverTrackBallAxis; }

	int fboRenderedIndex(QPoint p)const;
	void getModelBound(ldp::Float3& bmin, ldp::Float3& bmax)const;

	void setSmplMode(bool s) { m_isSmplMode = s; }
	bool isSmplMode()const { return m_isSmplMode; }
protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mouseDoubleClickEvent(QMouseEvent *ev);
	void wheelEvent(QWheelEvent*);
	void keyPressEvent(QKeyEvent*);
	void keyReleaseEvent(QKeyEvent*);
	void renderSelectionOnFbo();
	void renderDragBox();
	void renderTrackBall(bool idxMode);
	void timerEvent(QTimerEvent* ev);
	void renderMeshForSelection();
	void renderStitches();
	void renderGroupPlane();

protected:
	CShaderManager m_shaderManager;
	GLuint m_phong_program;
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType;
	Qt::MouseButtons m_buttons;
	QGLFramebufferObject* m_fbo;
	QImage m_fboImage;
	bool m_isDragBox;
	QPoint m_dragBoxBegin;
	TrackBallMode m_trackBallMode;
	ldp::Float3 m_trackBallPos;
	ldp::Mat3f m_trackBallR;
	float m_trackBallScale;
	int m_activeTrackBallAxis;
	int m_hoverTrackBallAxis;
	bool m_isSmplMode;
	Abstract3dEventHandle* m_currentEventHandle;
	std::vector<std::shared_ptr<Abstract3dEventHandle>> m_eventHandles;

	ldp::ClothManager* m_clothManager;
	ClothDesigner* m_mainUI;

protected:
	GLuint m_shadowDepthTexture = 0;
	GLuint m_shadowDepthFbo = 0;
	ldp::Float3 m_lightPosition;
	void initializeShadowMap();
	void renderShadowMap();
};

