#include "IdxPool.h"

namespace ldp
{
	std::hash_set<size_t> IdxPool::m_usedIdxCount;
	size_t IdxPool::m_nextIdx = 1;
	bool IdxPool::m_disableInc = false;
}