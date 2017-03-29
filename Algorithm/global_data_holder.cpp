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
	m_exportSepMesh = true;
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
		else if (lineLabel == "smpl_shape_dir")
			m_lastSmplShapeCoeffDir = lineBuffer;
		else if (lineLabel == "cloth_mesh_dir")
			m_lastClothMeshDir = lineBuffer;
		else if (lineLabel == "cloth_mesh_script_dir")
			m_lastClothMeshRenderScriptDir = lineBuffer;
		else if (lineLabel == "export_separated_mesh")
			m_exportSepMesh = !!atoi(lineBuffer.c_str());
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
	stm << "smpl_shape_dir: " << m_lastSmplShapeCoeffDir << std::endl;
	stm << "cloth_mesh_dir: " << m_lastClothMeshDir << std::endl;
	stm << "cloth_mesh_script_dir: " << m_lastClothMeshRenderScriptDir << std::endl;
	stm << "export_separated_mesh: " << int(m_exportSepMesh) << std::endl;
	stm.close();
}

void GlobalDataHolder::loadSvg(std::string name)
{
	m_lastSvgDir = name;
	m_clothManager->clear();
	m_clothManager->loadPiecesFromSvg(name.c_str());
	m_clothManager->simulationInit();
}


