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
#include "BatchSimulateManager.h"
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
	void dragEnterEvent(QDragEnterEvent* ev);
	void dropEvent(QDropEvent* ev);
	void closeEvent(QCloseEvent* ev);

	void loadSvg(QString name);
	void loadProjectXml(QString name);

	void updateUiByParam();
	Viewer2d* viewer2d() { return m_widget2d; }
	Viewer3d* viewer3d() { return m_widget3d; }

	void pushHistory(QString name, ldp::HistoryStack::Type type);
	public slots:
	void on_actionLoad_project_triggered();
	void on_actionSave_project_triggered();
	void on_actionSave_as_triggered();
	void on_actionLoad_svg_triggered();
	void on_actionExport_body_mesh_triggered();
	void on_actionExport_cloth_mesh_triggered();
	void on_actionExport_batch_simulation_triggered();
	void on_actionPlace_3d_by_2d_triggered();
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
	void on_cbSelfCollision_clicked();
	///
	void on_pbFlipPolygon_clicked();
	void on_sbDparamTriangleSize_valueChanged(double v);
	void on_dbPieceBendMult_valueChanged(double v);
	void on_dbPieceOutgoDist_valueChanged(double v);
	void on_pbMirrorSelected_clicked();
	void on_pbCopySelected_clicked();
public:
	Ui::ClothDesignerClass ui;
	Viewer2d* m_widget2d;
	Viewer3d* m_widget3d;
	int m_simulateTimer;
	int m_fpsTimer;
	int m_batchSimulateTimer;

	//////////////////////////////////////////////////////////////////////////
protected:
	QSplitter* m_splitter;
	void init3dActions();
	void add3dButton(Abstract3dEventHandle::ProcessorType type);
	void init2dActions();
	void add2dButton(Abstract2dEventHandle::ProcessorType type);
	public slots:
	void on_mainToolBar_actionTriggered(QAction* action);
	/////////////////////////////////////////////////////////////////////////
protected:
	QVector<QSharedPointer<QSlider>> m_smplShapeSliders;
	bool m_sliderEnableSmplUpdate = true;
	void setupSmplUI();
	void updateSmplUI();
	void saveProject(const std::string& fileName);
	void saveProjectAs();
	void exportClothMesh(const std::string& name);
	bool bindClothesToSmpl();
	void simulateCloth(int iterNum);
	void updateBodyState();
	void initBatchSimulation(QStringList* patternPaths);
	void finishBatchSimForCurPattern();
	void initBatchSimForCurBody();
	void initBatchSimForCurPattern(QString patternName);
	void updateBodyForBatchSimulation();
	void updateShapeForBatchSimulation();
	void updatePoseForBatchSimulation();
	void recordDataForBatchSimulation();
	void resetSmpl();
	public slots:
	void on_pbSaveSmplCoeffs_clicked();
	void on_pbLoadSmplCoeffs_clicked();
	void on_pbResetSmplCoeffs_clicked();
	void on_pbBindClothesToSmpl_clicked();
	void on_pbLoadSmplFromXml_clicked();
	void onSmplShapeSlidersValueChanged(int v);
private:
	bool m_projectSaved;
	BatchSimulateManager m_batchSimManager;
};

#endif // CLOTHDESIGNER_H
