#include "IdxPool.h"

namespace ldp
{
	std::hash_map<size_t, size_t> IdxPool::m_usedIdxCount;
	size_t IdxPool::m_nextIdx = 1;
	bool IdxPool::m_disableInc = false;
}