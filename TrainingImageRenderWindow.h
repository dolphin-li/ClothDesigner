#pragma once

#include <QtWidgets/QMainWindow>
#include "Ui_TrainingImageRenderWindow.h"
#include <memory>
class RenderedClothBodyInfo;
class ObjMesh;
class TrainingImageRenderWindow : public QMainWindow
{
	Q_OBJECT

public:
	TrainingImageRenderWindow(QWidget *parent = 0);
	~TrainingImageRenderWindow();

	void init();

	void loadBodyInfosFromXml(QString xmlName);
	RenderedClothBodyInfo* findCorrespondingBodyInfo(QString objFileName);
	public slots:
	void on_actionLoad_cloth_mesh_triggered();
	void on_actionBatch_render_dist_map_triggered();
	void on_actionRender_current_to_distmap_triggered();
protected:
	virtual void timerEvent(QTimerEvent* ev);
private:
	Ui_TrainingImageRenderWindow ui;
	std::shared_ptr<ObjMesh> m_clothMeshLoaded;
	std::vector<std::shared_ptr<RenderedClothBodyInfo>> m_bodyInfos;
};

