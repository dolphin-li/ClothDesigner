#pragma once

#include "ldputil.h"
#include <map>

namespace ldp
{
	class ClothManager;
	class LevelSet3D;
	class HistoryStack;
}
namespace arcsim
{
	class ArcSimManager;
}
class GlobalDataHolder
{
public:
	void init();

	void loadSvg(std::string name);

	void loadLastDirs();
	void saveLastDirs();
public:
	std::shared_ptr<ldp::ClothManager> m_clothManager;
	std::shared_ptr<ldp::HistoryStack> m_historyStack;
	std::shared_ptr<arcsim::ArcSimManager> m_arcsimManager;

	std::string m_lastSvgDir;
	std::string m_lastProXmlDir;
	std::string m_lastSmplShapeCoeffDir;
	std::string m_lastClothMeshDir;
	std::string m_lastClothMeshRenderScriptDir;

	bool m_exportSepMesh = true;
	bool m_arcsim_show_texcoord = false;
};

extern GlobalDataHolder g_dataholder;