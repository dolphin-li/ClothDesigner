#pragma once

#include "ldpMat\ldp_basic_vec.h"
#include "INTERSECTION.h"
#include "DISTANCE.h"
#include "HEAP.h"
class ObjMesh;
namespace ldp
{
	class LEVEL_SET_CELL_DATA;
	class LevelSet3D
	{
	public:
		typedef float ValueType;
		typedef ldp::ldp_basic_vec3<ValueType> Vec3;
	public:
		LevelSet3D();
		~LevelSet3D();

		void clear();
		void create(ldp::Int3 sz, Vec3 start, ValueType step);
		void setStartPos(Vec3 start);
		void fromMesh(const ObjMesh& mesh); // call create before this method
		void marchingCubeToMesh(ObjMesh& mesh)const;
		void load(std::string filename);
		void save(std::string filename)const;

		Vec3 getStartPos()const { return m_start; }
		ValueType getStep()const { return m_step; }
		ldp::Int3 size()const { return m_size; }
		int sizeXYZ()const { return (int)m_value.size(); }
		int sizeYZ()const { return m_size[1] * m_size[2]; }
		const ValueType* value()const { return m_value.data(); }
		ValueType* value() { return m_value.data(); }
		int index(int x, int y, int z)const { return x*sizeYZ() + y*m_size[2] + z; }
		const ValueType* value(int x, int y, int z)const { return m_value.data() + index(x, y, z); }
		ValueType* value(int x, int y, int z) { return m_value.data() + index(x, y, z); }
		const ValueType* value(ldp::Int3 idx)const { return value(idx[0], idx[1], idx[2]); }
		ValueType* value(ldp::Int3 idx) { return value(idx[0], idx[1], idx[2]); }

		ValueType localValue(ValueType x, ValueType y, ValueType z)const
		{
			if (x<2 || y<2 || z<2 || x>m_size[0] - 3 || y>m_size[1] - 3 || z>m_size[2] - 3)
				return MY_INFINITE;
			int i = std::floor(x), j = std::floor(y), k = std::floor(z);
			ValueType a = x - i, b = y - j, c = z - k;
			return	(1 - a)*(1 - b)*(1 - c)*value(i, j, k)[0] + a*(1 - b)*(1 - c)*value(i + 1, j, k)[0] +
				(1 - a)*(b)*(1 - c)*value(i, j + 1, k)[0] + a*(b)*(1 - c)*value(i + 1, j + 1, k)[0] +
				(1 - a)*(1 - b)*(c)*value(i, j, k + 1)[0] + a*(1 - b)*(c)*value(i + 1, j, k + 1)[0] +
				(1 - a)*(b)*(c)*value(i, j + 1, k + 1)[0] + a*(b)*(c)*value(i + 1, j + 1, k + 1)[0];
		}

		ValueType localValue(Vec3 p)const
		{
			return localValue(p[0], p[1], p[2]);
		}

		ValueType globalValue(ValueType x, ValueType y, ValueType z)const
		{
			return globalValue(Vec3(x, y, z));
		}

		ValueType globalValue(Vec3 p)const
		{
			return localValue((p-m_start)/m_step);
		}
	protected:
		void fastMarching(const int band_width = 6, bool reinitialize = true, bool boundary = false);
		void processCell(int i, int j, int k, HEAP <LEVEL_SET_CELL_DATA *> &heap_tree, 
			LevelSet3D::ValueType band_width, std::vector<LEVEL_SET_CELL_DATA>& c);
		ValueType computeExtendedDistance(int i, int j, int k, ValueType &sign,
			std::vector<LEVEL_SET_CELL_DATA>& c);
	private:
		ldp::Int3 m_size;
		Vec3 m_start;
		ValueType m_step;
		std::vector<ValueType> m_value;
	};
}