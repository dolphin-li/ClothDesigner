#include "ArcsimWindow.h"
#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "arcsim\ArcSimManager.h"
#include "Renderable\ObjMesh.h"
#include "tinyxml\tinyxml.h"
#include <QFileInfo>
#include <random>
#include <fstream>


ArcsimWindow::ArcsimWindow(QWidget *parent)
: QMainWindow(parent)
{
	ui.setupUi(this);
}

ArcsimWindow::~ArcsimWindow()
{

}

void ArcsimWindow::init()
{
	ui.widget->init(g_dataholder.m_arcsimManager.get());
}

void ArcsimWindow::timerEvent(QTimerEvent* ev)
{

}

void ArcsimWindow::on_actionLoad_conf_triggered()
{
	try
	{
		QString name = QFileDialog::getOpenFileName(this, "open arcsim config", "", "*.json");
		if (name.isEmpty())
			return;
		g_dataholder.m_arcsimManager->loadFromJsonConfig(name.toStdString());
		ui.widget->resetCamera();
		ui.widget->updateGL();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}

void ArcsimWindow::on_actionSave_cloth_triggered()
{

}

void ArcsimWindow::on_pbStartSimulation_clicked()
{

}