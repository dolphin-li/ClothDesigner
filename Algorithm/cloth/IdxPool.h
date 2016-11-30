#pragma once

#include <hash_map>
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
				auto& iter = m_usedIdxCount.find(oldIdx);
				if (iter == m_usedIdxCount.end())
					m_usedIdxCount.insert(std::make_pair(oldIdx, 1));
				else
					iter->second++;
				return oldIdx;
			}
			else
			{
				m_usedIdxCount.insert(std::make_pair(m_nextIdx, 1));
				return m_nextIdx++;
			}
		}
		static void freeIdx(size_t idx)
		{
			auto& iter = m_usedIdxCount.find(idx);
			if (iter == m_usedIdxCount.end())
				throw std::exception(std::string("IdxPool, freeIdx not existed: "
				+ std::to_string(idx)).c_str());
			iter->second--;
			if (iter->second == 0)
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
		static std::hash_map<size_t, size_t> m_usedIdxCount;
		static size_t m_nextIdx;
		static bool m_disableInc;
	};
}