#pragma once

#include <GL\glew.h>
#include <QtOpenGL>
#include "Camera\camera.h"
#include "event_handles\Abstract3dEventHandle.h"
#include "cloth\clothManager.h"
#include <Shader\ShaderManager.h>
class ClothDesigner;
class BatchSimulateManager;
class MeshRender;
class GPUBuffers;
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
	void simulateWheeling(int dir);
	TrackBallMode getTrackBallMode()const{ return m_trackBallMode; }
	void setActiveTrackBallAxis(int i){ m_activeTrackBallAxis = i; }
	int getActiveTrackBallAxis()const{ return m_activeTrackBallAxis; }
	void setHoverTrackBallAxis(int i){ m_hoverTrackBallAxis = i; }
	int getHoverTrackBallAxis()const{ return m_hoverTrackBallAxis; }
	void setBatchSimManager(BatchSimulateManager* bs){ m_batchSimManager = bs; };
	BatchSimulateManager* getBatchSimManager(){ return m_batchSimManager; }
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
	BatchSimulateManager* m_batchSimManager = nullptr;
	ldp::Camera m_camera;
	QPoint m_lastPos;
	int m_showType = 0;
	Qt::MouseButtons m_buttons = Qt::MouseButton::NoButton;
	QGLFramebufferObject* m_fbo = nullptr;
	QImage m_fboImage;
	bool m_isDragBox = false;
	QPoint m_dragBoxBegin;
	TrackBallMode m_trackBallMode = TrackBall_None;
	ldp::Float3 m_trackBallPos;
	ldp::Mat3f m_trackBallR;
	float m_trackBallScale = 1.f;
	int m_activeTrackBallAxis = 0;
	int m_hoverTrackBallAxis = 0;
	bool m_isSmplMode = false;
	Abstract3dEventHandle* m_currentEventHandle = nullptr;
	std::vector<std::shared_ptr<Abstract3dEventHandle>> m_eventHandles;

	ldp::ClothManager* m_clothManager = nullptr;
	ClothDesigner* m_mainUI = nullptr;
////////////////////////////////////////////////////////////////////////////////////
// shadow map related
protected:
	enum{
		MAX_LIGHTS_PASS = 4
	};
	int m_lightNum = 0;
	bool m_showShadow = false;
	bool m_shadowMapInitialized = false;
	bool m_showSubdiv = false;
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

