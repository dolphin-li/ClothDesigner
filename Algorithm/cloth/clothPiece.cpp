#include "clothPiece.h"

namespace ldp
{
	std::set<std::string> ClothPiece::s_nameSet;

	ClothPiece::ClothPiece()
	{
		m_name = generateUniqueName("default");
	}

	ClothPiece::~ClothPiece()
	{
	
	}

	std::string ClothPiece::setName(std::string name)
	{
		m_name = generateUniqueName(name);
		return m_name;
	}

	std::string ClothPiece::generateUniqueName(std::string nameHints)
	{
		auto iter = s_nameSet.find(nameHints);
		if (iter == s_nameSet.end())
		{
			s_nameSet.insert(nameHints);
			return nameHints;
		}

		for (int k = 0; k < 999999; k++)
		{
			std::string nm = nameHints + "_" + std::to_string(k);
			auto iter = s_nameSet.find(nm);
			if (iter == s_nameSet.end())
			{
				s_nameSet.insert(nm);
				return nm;
			}
		} // end for k
		throw std::exception("ClothPiece : cannot generate a unique name!");
	}
}