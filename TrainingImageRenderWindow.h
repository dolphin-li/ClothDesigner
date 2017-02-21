#pragma once

#include <QtWidgets/QMainWindow>
#include "Ui_TrainingImageRenderWindow.h"
#include <memory>
class ObjMesh;
class TrainingImageRenderWindow : public QMainWindow
{
	Q_OBJECT

public:
	TrainingImageRenderWindow(QWidget *parent = 0);
	~TrainingImageRenderWindow();

	void init();

	public slots:
	void on_actionLoad_cloth_mesh_triggered();
protected:
	virtual void timerEvent(QTimerEvent* ev);
private:
	Ui_TrainingImageRenderWindow ui;
	std::shared_ptr<ObjMesh> m_clothMeshLoaded;
};

