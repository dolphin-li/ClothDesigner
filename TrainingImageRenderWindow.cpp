#include "TrainingImageRenderWindow.h"
#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "tinyxml\tinyxml.h"
#include <QFileInfo>

struct RenderedClothBodyInfo
{
	std::string clothPath;
	std::string posePath;
	std::vector<float> shape;
	std::vector<float> pose;

	void load(TiXmlElement* bodyNode)
	{
		clothPath = "";
		posePath = "";
		shape.clear();
		pose.clear();
		for (auto ele = bodyNode->FirstChildElement(); ele; ele = ele->NextSiblingElement())
		{
			if (ele->Value() == std::string("ClothPath"))
				clothPath = ele->GetText();
			else if (ele->Value() == std::string("PosePath"))
				posePath = ele->GetText();
			else if (ele->Value() == std::string("shape"))
			{
				std::stringstream stm(ele->Attribute("value"));
				while (!stm.eof())
				{
					float val = 0.f;
					stm >> val;
					shape.push_back(val);
				}
			} // end if shape
			else if (ele->Value() == std::string("pose"))
			{
				std::stringstream stm(ele->Attribute("value"));
				while (!stm.eof())
				{
					float val = 0.f;
					stm >> val;
					pose.push_back(val);
				}
			} // end if pose
		} // end for ele
	}
};

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

		// load obj
		m_clothMeshLoaded->loadObj(name.toStdString().c_str(), true, false);

		// load corresponding xml
		QFileInfo finfo(name);
		loadBodyInfosFromXml(QDir::cleanPath(finfo.absoluteDir().absolutePath() + QDir::separator() + "Bodyinfo.xml"));

		// update body mesh
		if (g_dataholder.m_clothManager.get())
		{
			auto bodyInfo = findCorrespondingBodyInfo(name);
			if (bodyInfo)
			{
				auto smpl = g_dataholder.m_clothManager->bodySmplManager();
				smpl->setPoseShapeVals(&bodyInfo->pose, &bodyInfo->shape);
			} // end if bodyInfo
			g_dataholder.m_clothManager->updateSmplBody();
		} // end update body mesh

		// initialize widget
		ui.widget->init(g_dataholder.m_clothManager.get(), m_clothMeshLoaded.get());
		ui.widget->resetCamera();
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

void TrainingImageRenderWindow::loadBodyInfosFromXml(QString xmlName)
{
	TiXmlDocument doc;
	doc.LoadFile(xmlName.toStdString().c_str());
	if (doc.Error())
		throw std::exception(doc.ErrorDesc());

	auto root = doc.FirstChildElement();
	if (root == nullptr)
		throw std::exception("null xml document!");
	if (root->Value() != std::string("BodyInfoDocument"))
		throw std::exception("unsupported document, root must be \"BodyInfoDocument\"!");

	m_bodyInfos.clear();
	for (auto bodyNode = root->FirstChildElement(); bodyNode; bodyNode = bodyNode->NextSiblingElement())
	{
		m_bodyInfos.push_back(std::shared_ptr<RenderedClothBodyInfo>(new RenderedClothBodyInfo()));
		m_bodyInfos.back()->load(bodyNode);
	} // end for bodyNode
}

RenderedClothBodyInfo* TrainingImageRenderWindow::findCorrespondingBodyInfo(QString objFileName)
{
	QFileInfo objInfo(objFileName);
	QString pureObjName = objInfo.baseName();

	for (auto& bodyInfo : m_bodyInfos)
	{
		QFileInfo bif = bodyInfo->clothPath.c_str();
		if (pureObjName == bif.baseName())
			return bodyInfo.get();
	}
	return nullptr;
}