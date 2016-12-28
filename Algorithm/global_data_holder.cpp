#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "cloth\LevelSet3D.h"
#include "cloth\HistoryStack.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include <fstream>
GlobalDataHolder g_dataholder;

void GlobalDataHolder::init()
{
	m_clothManager.reset(new ldp::ClothManager);
	m_historyStack.reset(new ldp::HistoryStack);
	loadLastDirs();
}

void GlobalDataHolder::loadLastDirs()
{
	std::ifstream stream("__lastinfo.txt");
	if (stream.fail())
		return;
	std::string lineLabel, lineBuffer;
	while (!stream.eof())
	{
		std::getline(stream, lineBuffer);
		if (lineBuffer[0] == '#')
			continue;

		std::string lineLabel = ldp::getLineLabel(lineBuffer);
		if (lineLabel == "svg_dir")
			m_lastSvgDir = lineBuffer;
		else if (lineLabel == "proj_dir")
			m_lastProXmlDir = lineBuffer;
	}
	stream.close();
}

void GlobalDataHolder::saveLastDirs()
{
	std::ofstream stm("__lastinfo.txt");
	if (stm.fail())
		return;
	stm << "svg_dir: " << m_lastSvgDir << std::endl;
	stm << "proj_dir: " << m_lastProXmlDir << std::endl;
	stm.close();
}

void GlobalDataHolder::loadSvg(std::string name)
{
	m_lastSvgDir = name;
	m_clothManager->clear();
	m_clothManager->loadPiecesFromSvg(name.c_str());
	m_clothManager->bodyMeshInit()->loadObj("data/wm2_15k.obj", true, false);

	auto body = m_clothManager->bodyMeshInit();
	ldp::TransformInfo info;
	info.setIdentity();
	info.translate((body->boundingBox[0] + body->boundingBox[1])*-0.5f);
	info.rotate(ldp::QuaternionF().fromAngles(ldp::Float3(ldp::PI_S / 2, 0, ldp::PI_S)).toRotationMatrix3(), 0);
	info.scale(2.619848, 0);
	m_clothManager->setBodyMeshTransform(info);

	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;

	m_clothManager->simulationInit();
}


