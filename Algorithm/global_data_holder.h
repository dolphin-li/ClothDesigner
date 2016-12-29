#pragma once

#include "ldputil.h"
#include <map>

namespace ldp
{
	class ClothManager;
	class LevelSet3D;
	class HistoryStack;
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

	std::string m_lastSvgDir;
	std::string m_lastProXmlDir;
	std::string m_lastSmplShapeCoeffDir;
};

extern GlobalDataHolder g_dataholder;