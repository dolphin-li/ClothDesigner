#ifndef CLOTHDESIGNER_H
#define CLOTHDESIGNER_H

#include <QtWidgets/QMainWindow>
#include "ui_clothdesigner.h"
#include <QSplitter>
#include <QPushButton>
#include <QSharedPointer>
#include <QSignalMapper>
#include "ldpMat\ldp_basic_mat.h"
#include "event_handles\Abstract3dEventHandle.h"
#include "event_handles\Abstract2dEventHandle.h"
#include "cloth\HistoryStack.h"
class Viewer3d;
class Viewer2d;
class ClothDesigner : public QMainWindow
{
	Q_OBJECT

public:
	ClothDesigner(QWidget *parent = 0);
	~ClothDesigner();

	void timerEvent(QTimerEvent* ev);
	virtual void resizeEvent(QResizeEvent* ev);

	void updateUiByParam();
	Viewer2d* viewer2d() { return m_widget2d; }
	Viewer3d* viewer3d() { return m_widget3d; }

	void pushHistory(QString name, ldp::HistoryStack::Type type);
	public slots:
	void on_actionLoad_project_triggered();
	void on_actionSave_project_triggered();
	void on_actionLoad_svg_triggered();
	void on_actionPrev_triggered();
	void on_actionNext_triggered();
	void on_pbResetSimulation_clicked();
	void on_sbSparamOuterIter_valueChanged(int v);
	void on_sbSparamInnerIter_valueChanged(int v);
	void on_sbSparamTimeStepInv_valueChanged(int v);
	void on_sbSparamLapDampIter_valueChanged(int v);
	void on_sbSparamAirDamp_valueChanged(double v);
	void on_sbSparamControlStiff_valueChanged(double v);
	void on_sbSparamRho_valueChanged(double v);
	void on_sbSparamUnderRelax_valueChanged(double v);
	void on_sbSparamSpringStiff_valueChanged(double v);
	void on_sbSparamBendStiff_valueChanged(double v);
	void on_sbSparamStitchStiff_valueChanged(double v);
	void on_sbSparamStitchSpeed_valueChanged(double v);
	void on_sbSparamStitchBend_valueChanged(double v);
	void on_sbSparamGravityX_valueChanged(double v);
	void on_sbSparamGravityY_valueChanged(double v);
	void on_sbSparamGravityZ_valueChanged(double v);
	///
	void on_pbFlipPolygon_clicked();
	void on_sbDparamTriangleSize_valueChanged(double v);
private:
	Ui::ClothDesignerClass ui;
	Viewer2d* m_widget2d;
	Viewer3d* m_widget3d;
	int m_simulateTimer;
	int m_fpsTimer;
	//////////////////////////////////////////////////////////////////////////
protected:
	QSplitter* m_splitter;
	void init3dActions();
	void add3dButton(Abstract3dEventHandle::ProcessorType type);
	void init2dActions();
	void add2dButton(Abstract2dEventHandle::ProcessorType type);
	public slots:
	void on_mainToolBar_actionTriggered(QAction* action);
};

#endif // CLOTHDESIGNER_H
