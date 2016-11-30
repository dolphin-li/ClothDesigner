#pragma once

#include <hash_set>
#include <exception>
#include <string>
#include <algorithm>
#include <stdint.h>
namespace ldp
{
	// IdxPool: maintain a global index pool
	//	if enableIdxIncrement() [default], then each index is uniquely generated
	//	if disableIdxIncrement(), then eachIdx is added as a reference count.
	class IdxPool
	{
	public:
		static void clear()
		{
			m_usedIdxCount.clear();
			m_nextIdx = 1;
		}
		static size_t requireIdx(size_t oldIdx)
		{
			if (m_nextIdx >= UINT64_MAX - 1)
			{
				printf("IdxPool, warning: index out of range!\n");
			}

			if (m_disableInc)
			{
				m_nextIdx = std::max(oldIdx + 1, m_nextIdx);
				m_usedIdxCount.insert(oldIdx);
				return oldIdx;
			}
			else
			{
				m_usedIdxCount.insert(m_nextIdx);
				return m_nextIdx++;
			}
		}
		static void freeIdx(size_t idx)
		{
			auto& iter = m_usedIdxCount.find(idx);
			if (iter == m_usedIdxCount.end())
				throw std::exception(std::string("IdxPool, freeIdx not existed: "
				+ std::to_string(idx)).c_str());
			m_usedIdxCount.erase(iter);
		}
		static void disableIdxIncrement()
		{
			m_disableInc = true;
		}
		static void enableIdxIncrement()
		{
			m_disableInc = false;
		}
	protected:
		static std::hash_set<size_t> m_usedIdxCount;
		static size_t m_nextIdx;
		static bool m_disableInc;
	};
}