#pragma once

#include <QtWidgets/QMainWindow>
#include "Ui_ArcsimWindow.h"
#include <memory>
class ObjMesh;
class ArcsimWindow : public QMainWindow
{
	Q_OBJECT

public:
	ArcsimWindow(QWidget *parent = 0);
	~ArcsimWindow();

	void init();

	public slots:
	void on_actionLoad_conf_triggered();
	void on_actionSave_cloth_triggered();
protected:
	virtual void timerEvent(QTimerEvent* ev);
private:
	Ui_ArcsimWindow ui;
};

