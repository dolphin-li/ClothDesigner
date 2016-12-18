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
public:
	std::shared_ptr<ldp::ClothManager> m_clothManager;
	std::shared_ptr<ldp::HistoryStack> m_historyStack;
};

extern GlobalDataHolder g_dataholder;