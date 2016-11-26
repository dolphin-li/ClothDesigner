#pragma once

#include "ldputil.h"
#include <map>

namespace ldp
{
	class ClothManager;
	class LevelSet3D;
}
class GlobalDataHolder
{
public:
	void init();

	void debug_1();
	void debug_2();
	void debug_3();
public:
	std::shared_ptr<ldp::ClothManager> m_clothManager;
};

extern GlobalDataHolder g_dataholder;