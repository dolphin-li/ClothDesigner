#include "TrainingImageRenderWindow.h"
#include "global_data_holder.h"
#include "Renderable\ObjMesh.h"
TrainingImageRenderWindow::TrainingImageRenderWindow(QWidget *parent)
: QMainWindow(parent)
{
	ui.setupUi(this);
	m_clothMeshLoaded.reset(new ObjMesh);
}

TrainingImageRenderWindow::~TrainingImageRenderWindow()
{

}

void TrainingImageRenderWindow::init()
{
	ui.widget->init(g_dataholder.m_clothManager.get(), m_clothMeshLoaded.get());
}

void TrainingImageRenderWindow::on_actionLoad_cloth_mesh_triggered()
{
	try
	{
		QString name = QFileDialog::getOpenFileName(this, "load cloth mesh", 
			g_dataholder.m_lastClothMeshDir.c_str(), "*.obj");
		if (name.isEmpty())
			return;
		g_dataholder.m_lastClothMeshDir = name.toStdString();
		g_dataholder.saveLastDirs();

		m_clothMeshLoaded->loadObj(name.toStdString().c_str(), true, false);

		ui.widget->init(g_dataholder.m_clothManager.get(), m_clothMeshLoaded.get());
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error " << std::endl;
	}
}

void TrainingImageRenderWindow::timerEvent(QTimerEvent* ev)
{

}