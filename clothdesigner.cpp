#include "clothdesigner.h"
#include <QGridLayout>
#include "global_data_holder.h"

#include "viewer3d.h"

ClothDesigner::ClothDesigner(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	
	ui.centralWidget->setLayout(new QGridLayout());
	m_splitter = new QSplitter(ui.centralWidget);
	ui.centralWidget->layout()->addWidget(m_splitter);
	m_widget2d = new QWidget(m_splitter);
	m_widget3d = new Viewer3d(m_splitter);
	m_splitter->addWidget(m_widget3d);
	m_splitter->addWidget(m_widget2d);

	g_dataholder.init();
	m_widget3d->init(g_dataholder.m_clothManager.get());
}

ClothDesigner::~ClothDesigner()
{

}
