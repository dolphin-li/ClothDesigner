#include "TrainingImageRenderWindow.h"
#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "tinyxml\tinyxml.h"
#include <QFileInfo>
#include <random>
#include <fstream>

#pragma region --random number generator

static std::random_device g_batch_rd;
static std::mt19937 g_batch_rgen(g_batch_rd());
static void batch_rand_reset()
{
	g_batch_rgen.seed(1234);
}
static int batch_rand()
{
	std::uniform_int_distribution<> dist;
	return dist(g_batch_rgen);
}

static float batch_rand_norm(float mean = 0, float var = 1)
{
	std::normal_distribution<float> dist(mean, var);
	return dist(g_batch_rgen);
}
#pragma endregion

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

struct BatchRenderDistMap
{
	int timerId = 0;
	QString renderType;
	QString root_folder;
	QString bodyTrans_xml;
	float camera_up_down_degree = 0;
	float camera_left_right_degree = 0;
	float camera_near_far_scale = 1;
	int random_views = 1;

	TrainingImageRenderWindow* window = nullptr;
	int currentClothMeshId = 0;
	int currentViewId = 0;
	QVector<QString> clothSizeLabels;
	QVector<QString> clothRenderIds;
	std::vector<QMap<QString, std::shared_ptr<RenderedClothBodyInfo>>> m_bodyInfos;

	BatchRenderDistMap(TrainingImageRenderWindow* w) :window(w) {}

	void reset()
	{
		timerId = 0;
		renderType = "";
		root_folder = "";
		camera_up_down_degree = 0;
		camera_left_right_degree = 0;
		camera_near_far_scale = 1;
		random_views = 1;
		clothSizeLabels.clear();
		clothRenderIds.clear();
		currentClothMeshId = 0;
		currentViewId = 0;
	}

	void loadConfig(QString filename)
	{
		reset();
		std::string lineBuffer, lineLabel;
		std::ifstream stream(filename.toStdString());
		if (stream.fail())
			throw std::exception(("IOError: " + filename).toStdString().c_str());
		QFileInfo finfo(filename);
		while (!stream.eof())
		{
			std::getline(stream, lineBuffer);
			if (lineBuffer[0] == '#')
				continue;

			std::string lineLabel = ldp::getLineLabel(lineBuffer);
			if (lineLabel == "root_folder")
				root_folder = QDir::cleanPath(finfo.absolutePath() + QDir::separator() + lineBuffer.c_str());
			else if (lineLabel == "bodyTrans_xml")
				bodyTrans_xml = lineBuffer.c_str();
			else if (lineLabel == "render_type")
				renderType = lineBuffer.c_str();
			else if (lineLabel == "camera_up_down_degree")
				camera_up_down_degree = QString(lineBuffer.c_str()).toDouble();
			else if (lineLabel == "camera_left_right_degree")
				camera_left_right_degree = QString(lineBuffer.c_str()).toDouble();
			else if (lineLabel == "camera_near_far_scale")
				camera_near_far_scale = QString(lineBuffer.c_str()).toDouble();
			else if (lineLabel == "random_views")
				random_views = QString(lineBuffer.c_str()).toInt();
		}
		stream.close();
	}

	void prepareData()
	{
		currentClothMeshId = 0;

		// 1. get all size labels
		QDirIterator rootDirIter(root_folder);
		clothSizeLabels.clear();
		while (rootDirIter.hasNext())
		{
			QFileInfo finfo(rootDirIter.next());
			if (finfo.baseName() == "." || finfo.baseName() == ".."
				|| finfo.baseName().isEmpty() || !finfo.isDir())
				continue;
			clothSizeLabels.push_back(finfo.baseName());
		} // end while rootDirIter

		if (clothSizeLabels.isEmpty())
			return;

		// 2. get all rendered meshes in each size
		QDirIterator labelDirIter(QDir::cleanPath(root_folder + QDir::separator() + clothSizeLabels[0]));
		clothRenderIds.clear();
		QSet<QString> idSet;
		while (labelDirIter.hasNext())
		{
			QFileInfo finfo(labelDirIter.next());
			if (finfo.baseName().isEmpty() || finfo.isDir() || 
				!finfo.fileName().toLower().endsWith(".obj"))
				continue;
			clothRenderIds.push_back(finfo.baseName());
			idSet.insert(finfo.baseName());
		} // end while labelDirIter

		// 3. check data and load xml info
		m_bodyInfos.clear();
		for (const auto& label : clothSizeLabels)
		{
			QDir curDir(QDir::cleanPath(root_folder + QDir::separator() + clothSizeLabels[0]));

			// load xml
			QString xmlName(QDir::cleanPath(curDir.absolutePath() + QDir::separator() + "BodyInfo.xml"));
			std::vector<std::shared_ptr<RenderedClothBodyInfo>> tmpBodyInfos;
			window->loadBodyInfosFromXml(xmlName, tmpBodyInfos);
			QMap<QString, std::shared_ptr<RenderedClothBodyInfo>> tmpMap;
			for (const auto& info : tmpBodyInfos)
			{
				QFileInfo finfo(info->clothPath.c_str());
				tmpMap[finfo.baseName()] = info;
			}
			m_bodyInfos.push_back(tmpMap);

			// check id
			QDirIterator curDirIter(curDir);
			QSet<QString> curIdSet;
			while (curDirIter.hasNext())
			{
				QFileInfo finfo(curDirIter.next());
				if (finfo.baseName().isEmpty() || finfo.isDir() ||
					!finfo.fileName().toLower().endsWith(".obj"))
					continue;
				if (idSet.find(finfo.baseName()) == idSet.end())
					throw std::exception(QString().sprintf("Error: id %s/%s not found in %s", 
					label.toStdString().c_str(), 
					finfo.baseName().toStdString().c_str(), 
					clothSizeLabels[0].toStdString().c_str()
					).toStdString().c_str());
				curIdSet.insert(finfo.baseName());
			} // end while labelDirIter

			for (auto id : idSet)
			{
				if (curIdSet.find(id) == curIdSet.end())
					throw std::exception(QString().sprintf("Error: id %s/%s not found in %s",
					clothSizeLabels[0].toStdString().c_str(), 
					id.toStdString().c_str(), 
					label.toStdString().c_str()
					).toStdString().c_str());
			}
		} // end for iLabel
	}

	void showConfig()
	{
		printf("##################batch render cloth to dist map config\n");
		printf("bodyTrans_xml: %s\n", bodyTrans_xml.toStdString().c_str());
		printf("root_folder: %s\n", root_folder.toStdString().c_str());
		printf("renderType: %s\n", renderType.toStdString().c_str());
		printf("camera_up_down_degree: %f\n", camera_left_right_degree);
		printf("camera_left_right_degree: %f\n", camera_left_right_degree);
		printf("camera_near_far_scale: %f\n", camera_near_far_scale);
	}

	void randomCamera(ldp::Camera& cam)
	{
		auto loc = cam.getLocation();
		auto tar = cam.getDirection() + cam.getLocation();
		auto up = cam.getUp();

		float scale = batch_rand_norm(1, (camera_near_far_scale - 1)/2);
		float xrot = batch_rand_norm(0, camera_left_right_degree * ldp::PI_S / 180 / 2);
		float yrot = batch_rand_norm(0, camera_up_down_degree* ldp::PI_S / 180 / 2);

		auto dir = tar - loc;
		ldp::QuaternionF qx, qy;
		qx.fromAngleAxis(xrot, up.cross(dir).normalize());
		qy.fromAngleAxis(yrot, up.normalize());
		dir = qy.applyVec(qx.applyVec(dir)) * scale;
		loc = tar - dir;

		cam.lookAt(loc, tar, up);
	}
};

TrainingImageRenderWindow::TrainingImageRenderWindow(QWidget *parent)
: QMainWindow(parent)
{
	ui.setupUi(this);
	m_clothMeshLoaded.reset(new ObjMesh);
	m_batchDistMapRenderer.reset(new BatchRenderDistMap(this));
}

TrainingImageRenderWindow::~TrainingImageRenderWindow()
{

}

void TrainingImageRenderWindow::init()
{
	ui.widget->init(g_dataholder.m_clothManager.get(), m_clothMeshLoaded.get());
}

void TrainingImageRenderWindow::timerEvent(QTimerEvent* ev)
{
	if (ev->timerId() == m_batchDistMapRenderer->timerId)
	{
		// iterate over all sizes
		ui.widget->resetCamera();
		m_batchDistMapRenderer->randomCamera(ui.widget->camera());
		for (int iLabel = 0; iLabel < m_batchDistMapRenderer->clothSizeLabels.size(); iLabel++)
		{
			QString objName = QDir::cleanPath(m_batchDistMapRenderer->root_folder + QDir::separator()
				+ m_batchDistMapRenderer->clothSizeLabels[iLabel] + QDir::separator()
				+ m_batchDistMapRenderer->clothRenderIds[m_batchDistMapRenderer->currentClothMeshId]
				+ ".obj");
			QFileInfo oinfo(objName);
			const auto& bodyInfo = m_batchDistMapRenderer->m_bodyInfos[iLabel]
				[oinfo.baseName()];
			QFileInfo binfo(bodyInfo->clothPath.c_str());
			if (binfo.baseName() != oinfo.baseName())
				throw std::exception(QString().sprintf("%s/bodyinfo.xml not valid: %s-%s not matched!",
				oinfo.absolutePath().toStdString().c_str(), binfo.baseName().toStdString().c_str(),
				oinfo.baseName().toStdString().c_str()).toStdString().c_str());

			// load objMesh
			m_clothMeshLoaded->loadObj(objName.toStdString().c_str(), true, false);

			// update body mesh
			auto smpl = g_dataholder.m_clothManager->bodySmplManager();
			smpl->setPoseShapeVals(&bodyInfo->pose, &bodyInfo->shape);
			g_dataholder.m_clothManager->updateSmplBody();

			// render
			ui.widget->updateGL();
			std::vector<QImage> imgs;
			ui.widget->generateDistMap_x9(imgs);
			for (size_t iMap = 0; iMap < imgs.size(); iMap++)
			{
				QString imgName = QDir::cleanPath(oinfo.absolutePath() + QDir::separator()
					+ oinfo.baseName() + QString().sprintf("_rv%d_%d.png", 
					m_batchDistMapRenderer->currentViewId, iMap));
				imgs[iMap].save(imgName);
			}
		} // end for iLabel
		printf("mesh_%d/view_%d/total_%d processed\n", m_batchDistMapRenderer->currentClothMeshId,
			m_batchDistMapRenderer->currentViewId,
			m_batchDistMapRenderer->clothRenderIds.size());

		// next data
		m_batchDistMapRenderer->currentViewId++;
		if (m_batchDistMapRenderer->currentViewId >= m_batchDistMapRenderer->random_views)
		{
			m_batchDistMapRenderer->currentViewId = 0;
			m_batchDistMapRenderer->currentClothMeshId++;
			if (m_batchDistMapRenderer->currentClothMeshId == m_batchDistMapRenderer->clothRenderIds.size())
			{
				killTimer(m_batchDistMapRenderer->timerId);
				printf("####################batch render dist map finished!");
			}
		}
	} // end if batch render timer id
}

void TrainingImageRenderWindow::on_actionBatch_render_dist_map_triggered()
{
	try
	{
		QString name = QFileDialog::getOpenFileName(this, "batch render dist map",
			g_dataholder.m_lastClothMeshRenderScriptDir.c_str(), "*.batch_render.txt");
		if (name.isEmpty())
			return;
		g_dataholder.m_lastClothMeshRenderScriptDir = name.toStdString();
		g_dataholder.saveLastDirs();
		m_batchDistMapRenderer->loadConfig(name);
		m_batchDistMapRenderer->showConfig();
		g_dataholder.m_clothManager->fromXml(m_batchDistMapRenderer->bodyTrans_xml.toStdString());
		m_batchDistMapRenderer->prepareData();
		ui.widget->init(g_dataholder.m_clothManager.get(), m_clothMeshLoaded.get());
		m_batchDistMapRenderer->timerId = startTimer(1000);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error " << std::endl;
	}
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
		loadBodyInfosFromXml(QDir::cleanPath(finfo.absoluteDir().absolutePath() 
			+ QDir::separator() + "Bodyinfo.xml"), m_bodyInfos);

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
		ui.widget->updateGL();
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error " << std::endl;
	}
}

void TrainingImageRenderWindow::on_actionRender_current_to_distmap_triggered()
{
	try
	{
		if (g_dataholder.m_lastClothMeshDir == "")
			return;
		ui.widget->updateGL();

		std::vector<QImage> imgs;
		ui.widget->generateDistMap_x9(imgs);

		QFileInfo finfo(g_dataholder.m_lastClothMeshDir.c_str());
		for (size_t iMap = 0; iMap < imgs.size(); iMap++)
		{
			QString imgName = QDir::cleanPath(finfo.absolutePath() + QDir::separator() 
				+ finfo.baseName() + QString().sprintf("_%d.png", iMap));
			imgs[iMap].save(imgName);
		}
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error " << std::endl;
	}
}

void TrainingImageRenderWindow::loadBodyInfosFromXml(QString xmlName, 
	std::vector<std::shared_ptr<RenderedClothBodyInfo>>& bodyInfos)const
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

	bodyInfos.clear();
	for (auto bodyNode = root->FirstChildElement(); bodyNode; bodyNode = bodyNode->NextSiblingElement())
	{
		bodyInfos.push_back(std::shared_ptr<RenderedClothBodyInfo>(new RenderedClothBodyInfo()));
		bodyInfos.back()->load(bodyNode);
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