#pragma once

#include "ldputil.h"
#include <map>

namespace ldp
{
	class ClothManager;
}
class GlobalDataHolder
{
public:
	void init();
public:
	std::shared_ptr<ldp::ClothManager> m_clothManager;
};

extern GlobalDataHolder g_dataholder;