#include "MaterialCache.h"
#include "arcsim\adaptiveCloth\conf.hpp"
#include "definations.h"
#include "ldputil.h"
namespace ldp
{
	template<class T, int n>
	inline ldp::ldp_basic_vec<T, n> convert(arcsim::Vec<n, T> v)
	{
		ldp::ldp_basic_vec<T, n> r;
		for (int k = 0; k < n; k++)
			r[k] = v[k];
		return r;
	}


	MaterialCache::MaterialCache()
	{
	}

	MaterialCache::~MaterialCache()
	{
	}

	void MaterialCache::clear()
	{
		m_data.clear();
	}

	const MaterialCache::Material* MaterialCache::findMaterial(std::string name)
	{
		auto iter = m_data.find(name);
		if (iter == m_data.end())
			return createMaterial(name);
		return iter->second.get();
	}

	const MaterialCache::Material* MaterialCache::createMaterial(std::string name)
	{
		std::shared_ptr<arcsim::Cloth::Material> mat(new arcsim::Cloth::Material);
		arcsim::load_material_data(*mat, ldp::fullfile(PieceParam::default_material_folder, name+".json"));

		std::shared_ptr<Material> material(new Material);

		// density
		material->density = mat->density;

		// stretch sample
		for (int x = 0; x < StretchingSamples::SAMPLES; x++)
		for (int y = 0; y < StretchingSamples::SAMPLES; y++)
		for (int z = 0; z < StretchingSamples::SAMPLES; z++)
			material->stretchSample(x, y, z) = convert(mat->stretching.s[x][y][z]);
		material->stretchSample.updateHostToDevice();

		// bend data
		for (int x = 0; x < material->bendData.cols(); x++)
		{
			int wrap_x = x;
			if (wrap_x>4)
				wrap_x = 8 - wrap_x;
			if (wrap_x > 2)
				wrap_x = 4 - wrap_x;
			for (int y = 0; y < material->bendData.rows(); y++)
				material->bendData(x, y) = mat->bending.d[wrap_x][y];
		}
		material->bendData.updateHostToDevice();

		// insert
		m_data.insert(std::make_pair(name, material));

		return material.get();
	}
}