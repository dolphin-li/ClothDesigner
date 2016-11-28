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
class Viewer3d;
class Viewer2d;
class ClothDesigner : public QMainWindow
{
	Q_OBJECT

public:
	ClothDesigner(QWidget *parent = 0);
	~ClothDesigner();

	void timerEvent(QTimerEvent* ev);

	void updateUiByParam();
	public slots:
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
	void on_sbSparamGravityX_valueChanged(double v);
	void on_sbSparamGravityY_valueChanged(double v);
	void on_sbSparamGravityZ_valueChanged(double v);
private:
	Ui::ClothDesignerClass ui;
	QSplitter* m_splitter;
	Viewer2d* m_widget2d;
	Viewer3d* m_widget3d;
	int m_simulateTimer;
	int m_fpsTimer;
	//////////////////////////////////////////////////////////////////////////
protected:
	QMap<Abstract3dEventHandle::ProcessorType, QSharedPointer<QPushButton>> m_leftDockButtons;
	QSharedPointer<QSignalMapper> m_ldbSignalMapper;
	void initLeftDockActions();
	void addLeftDockWidgetButton(Abstract3dEventHandle::ProcessorType type);
	public slots:
	void leftDocButtonsClicked(int i);
};

#endif // CLOTHDESIGNER_H
