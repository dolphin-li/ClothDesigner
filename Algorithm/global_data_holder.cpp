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
		else if (lineLabel == "smpl_shape_dir")
			m_lastSmplShapeCoeffDir = lineBuffer;
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
	stm.close();
}

void GlobalDataHolder::loadSvg(std::string name)
{
	m_lastSvgDir = name;
	m_clothManager->clear();
	m_clothManager->loadPiecesFromSvg(name.c_str());
	m_clothManager->simulationInit();
}


