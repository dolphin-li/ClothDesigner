#include "IdxPool.h"

namespace ldp
{
	std::hash_set<size_t> IdxPool::m_usedIdx;
	std::hash_set<size_t> IdxPool::m_freeIdx;
	size_t IdxPool::m_nextIdx = 1;
	bool IdxPool::m_disableInc = false;
}