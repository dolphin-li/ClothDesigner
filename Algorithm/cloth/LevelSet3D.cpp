#include "LevelSet3D.h"
#include <fstream>
#include "Renderable\ObjMesh.h"
#include "PROGRESSING_BAR.h"
namespace ldp
{
#pragma region --utils
//#define EXPORT_WANG_HUAMING_FORMAT
//#define IMPORT_WANG_HUAMING_FORMAT
	// Differencing schemes
#define UPWIND_DIFFERENCING		0
#define CENTRAL_DIFFERENCING	1
#define HJ_WENO_DIFFERENCING	2
	// #buffer cells to the boundary
#define BUFFER					4
	// Cell state for fast marching
#define UNPROCESSED				0
#define PROCESSING				1
#define PROCESSED				2
#define FOR_EVERY_CELL		\
	for (int i = 0, id = 0; i<m_size[0]; i++)	\
	for (int j = 0; j<m_size[1]; j++)			\
	for (int k = 0; k<m_size[2]; k++, id++)

	typedef LevelSet3D::ValueType ValueType;
	class LEVEL_SET_CELL_DATA
	{
	public:
		int		id;			// array index
		int		heap_node;	// heap node index	
		char	state;
	};

	inline void Process_Surface_Cell(int nid, int pid, 
		std::vector<LevelSet3D::ValueType>& temp,
		const std::vector<LevelSet3D::ValueType>& p, 
		std::vector<LEVEL_SET_CELL_DATA>& c)
	{
		if (p[pid] - p[nid]>1)
		{
			temp[pid] = Min_By_Abs(p[pid] / (p[pid] - p[nid]), temp[pid]);
			temp[nid] = Min_By_Abs(p[nid] / (p[pid] - p[nid]), temp[nid]);
		}
		c[pid].state = PROCESSED;
		c[nid].state = PROCESSED;
	}

	inline ValueType Extended_Distance_With_One_Value(const ValueType theta, const ValueType sign)
	{
		return theta + SIGN(sign);
	}

	inline ValueType Extended_Distance_With_Two_Value(const ValueType theta1, const ValueType theta2, const ValueType sign)
	{
		if (theta1 == MY_INFINITE) return Extended_Distance_With_One_Value(theta2, sign);
		if (theta2 == MY_INFINITE) return Extended_Distance_With_One_Value(theta1, sign);

		if (fabsf(theta1 - theta2)>1)
		{
			if (sign*theta1>sign*theta2) return Extended_Distance_With_One_Value(theta2, sign);
			else						return Extended_Distance_With_One_Value(theta1, sign);
		}
		ValueType a = 2;
		ValueType b = -2 * (theta1 + theta2);
		ValueType c = theta1*theta1 + theta2*theta2 - 1;
		return (-b + sign*sqrt(b*b - 4 * a*c)) / (2 * a);
	}

	inline LevelSet3D::ValueType Extended_Distance_With_Three_Value(const ValueType theta1, 
		const ValueType theta2, const ValueType theta3, const ValueType sign)
	{
		if (theta1 == MY_INFINITE) return Extended_Distance_With_Two_Value(theta2, theta3, sign);
		if (theta2 == MY_INFINITE) return Extended_Distance_With_Two_Value(theta1, theta3, sign);
		if (theta3 == MY_INFINITE) return Extended_Distance_With_Two_Value(theta1, theta2, sign);

		ValueType max_theta = Max(sign*theta1, sign*theta2, sign*theta3);
		if (SQR(max_theta - theta1) + SQR(max_theta - theta2) + SQR(max_theta - theta3)>1)
		{
			if (max_theta == sign*theta1)		return Extended_Distance_With_Two_Value(theta2, theta3, sign);
			else if (max_theta == sign*theta2) return Extended_Distance_With_Two_Value(theta1, theta3, sign);
			else if (max_theta == sign*theta3) return Extended_Distance_With_Two_Value(theta1, theta2, sign);
			else printf("Error in Extended_Distance_With_Three_Value?\n");
			return 0;
		}
		ValueType a = 3;
		ValueType b = -2 * (theta1 + theta2 + theta3);
		ValueType c = SQR(theta1) + SQR(theta2) + SQR(theta3) - 1;
		return (-b + sign*sqrt(b*b - 4 * a*c)) / (2 * a);
	}
#pragma endregion

	LevelSet3D::LevelSet3D()
	{
		m_step = ValueType(0);
	}

	LevelSet3D::~LevelSet3D()
	{
	
	}

	void LevelSet3D::clear()
	{
		m_value.clear();
		m_step = ValueType(0);
		m_size = ldp::Int3(0);
		m_start = Vec3(0);
	}

	void LevelSet3D::create(ldp::Int3 sz, Vec3 start, ValueType step)
	{
		clear();
		m_size = sz;
		m_start = start;
		m_step = step;
		m_value.resize(m_size[0] * m_size[1] * m_size[2], ValueType(MY_INFINITE));
	}

	void LevelSet3D::setStartPos(Vec3 start)
	{
		m_start = start;
	}

	void LevelSet3D::fromMesh(const ObjMesh& mesh)
	{
		// 2: Set up inside/outside
		std::vector<int> counters(m_value.size(), 0);
		int t = 0;
		for (const auto& f : mesh.face_list)
		{
			assert(f.vertex_count == 3);
			Vec3 v0 = mesh.vertex_list[f.vertex_index[0]];
			Vec3 v1 = mesh.vertex_list[f.vertex_index[1]];
			Vec3 v2 = mesh.vertex_list[f.vertex_index[2]];
			int min_ti = MAX(int((Min(v0[0], v1[0], v2[0]) - m_start[0]) / m_step) - 1, 0);
			int min_tj = MAX(int((Min(v0[1], v1[1], v2[1]) - m_start[1]) / m_step) - 1, 0);
			int max_ti = MIN(int((Max(v0[0], v1[0], v2[0]) - m_start[0]) / m_step) + 1, m_size[0] - 1);
			int max_tj = MIN(int((Max(v0[1], v1[1], v2[1]) - m_start[1]) / m_step) + 1, m_size[1] - 1);
			for (int i = min_ti; i <= max_ti; i++)
			for (int j = min_tj; j <= max_tj; j++)
			{
				Vec3 p0 = m_start + Vec3(i*m_step, j*m_step, -100);
				Vec3 dir(0, 0, 1);

				ValueType mint;
				if (Ray_Triangle_Intersection(v0.ptr(), v1.ptr(), v2.ptr(), p0.ptr(), dir.ptr(), mint))
				{
					mint = MAX((mint - 100) / m_step, 0);
					for (int k = (int)mint + 1; k<m_size[2]; k++) 
						counters[index(i, j, k)]++;
				}
			}
		}
		for (size_t i = 0; i<counters.size(); i++)
		{
			if (counters[i] % 2 == 0)	
				m_value[i] = 999999;
			else					
				m_value[i] = -999999;
		}
		counters.swap(std::vector<int>()); // force memory clean up

		// Part 3: Compute the distances
		PROGRESSING_BAR bar(mesh.face_list.size());
		for (const auto& f : mesh.face_list)
		{
			assert(f.vertex_count == 3);
			Vec3 v0 = mesh.vertex_list[f.vertex_index[0]];
			Vec3 v1 = mesh.vertex_list[f.vertex_index[1]];
			Vec3 v2 = mesh.vertex_list[f.vertex_index[2]];

			int min_ti = Max<int>((Min(v0[0], v1[0], v2[0]) - m_start[0]) / m_step - 4, 0);
			int min_tj = Max<int>((Min(v0[1], v1[1], v2[1]) - m_start[1]) / m_step - 4, 0);
			int min_tk = Max<int>((Min(v0[2], v1[2], v2[2]) - m_start[2]) / m_step - 4, 0);
			int max_ti = Min<int>((Max(v0[0], v1[0], v2[0]) - m_start[0]) / m_step + 4, m_size[0] - 1);
			int max_tj = Min<int>((Max(v0[1], v1[1], v2[1]) - m_start[1]) / m_step + 4, m_size[1] - 1);
			int max_tk = Min<int>((Max(v0[2], v1[2], v2[2]) - m_start[2]) / m_step + 4, m_size[2] - 1);
			for (int i = min_ti; i <= max_ti; i++)
			for (int j = min_tj; j <= max_tj; j++)
			for (int k = min_tk; k <= max_tk; k++)
			{
				Vec3 p0 = m_start + Vec3(i, j, k)*m_step;
				ValueType bb, bc;
				ValueType distance = sqrt(Squared_VT_Distance(p0.ptr(), v0.ptr(), 
					v1.ptr(), v2.ptr(), bb, bc)) / m_step;
				m_value[index(i, j, k)] = SIGN(m_value[index(i, j, k)])* Min(distance, fabs(m_value[index(i, j, k)]));
			}
			bar.Add();
		}
		bar.End();

		// Part 4: Propagate the signed distances
		fastMarching();
	}

	void LevelSet3D::marchingCubeToMesh(ObjMesh& mesh)const
	{
#pragma region --triTable
		const static char triTable[256][16] =
		{
			{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
			{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
			{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
			{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
			{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
			{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
			{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
			{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
			{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
			{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
			{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
			{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
			{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
			{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
			{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
			{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
			{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
			{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
			{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
			{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
			{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
			{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
			{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
			{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
			{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
			{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
			{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
			{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
			{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
			{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
			{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
			{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
			{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
			{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
			{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
			{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
			{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
			{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
			{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
			{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
			{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
			{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
			{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
			{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
			{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
			{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
			{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
			{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
			{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
			{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
			{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
			{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
			{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
			{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
			{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
			{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
			{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
			{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
			{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
			{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
			{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
			{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
			{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
			{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
			{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
			{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
			{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
			{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
			{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
			{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
			{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
			{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
			{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
			{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
			{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
			{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
			{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
			{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
			{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
			{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
			{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
			{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
			{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
			{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
			{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
			{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
			{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
			{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
			{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
			{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
			{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
			{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
			{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
			{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
			{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
			{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
			{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
			{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
			{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
			{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
			{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
			{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
			{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
			{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
			{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
			{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
			{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
			{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
			{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
			{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
		};
#pragma endregion

		std::vector<int> x_vertices(sizeXYZ(), 0);
		std::vector<int> y_vertices(sizeXYZ(), 0);
		std::vector<int> z_vertices(sizeXYZ(), 0);

		mesh.clear();

		FOR_EVERY_CELL
		{
			// Add x_vertices
			if (i + 1 != m_size[0])	
			if (m_value[index(i, j, k)] <= 0 && m_value[index(i + 1, j, k)] >= 0 || 
				m_value[index(i, j, k)] >= 0 && m_value[index(i + 1, j, k)] <= 0)
			{
				mesh.vertex_list.push_back(ldp::Float3(
					i + m_value[index(i, j, k)] / (m_value[index(i, j, k)] - m_value[index(i + 1, j, k)]), j, k)
					);
				x_vertices[index(i, j, k)] = mesh.vertex_list.size() - 1;
			}
			//Add y_vertices
			if (j + 1 != m_size[1])	
			if (m_value[index(i, j, k)] <= 0 && m_value[index(i, j + 1, k)] >= 0 ||
				m_value[index(i, j, k)] >= 0 && m_value[index(i, j + 1, k)] <= 0)
			{
				mesh.vertex_list.push_back(ldp::Float3(i, 
					j + m_value[index(i, j, k)] / (m_value[index(i, j, k)] - m_value[index(i, j + 1, k)]), k)
					);
				y_vertices[index(i, j, k)] = mesh.vertex_list.size() - 1;
			}
			//Add z_vertices
			if (k + 1 != m_size[2]) 
			if (m_value[index(i, j, k)] <= 0 && m_value[index(i, j, k + 1)] >= 0 ||
				m_value[index(i, j, k)] >= 0 && m_value[index(i, j, k + 1)] <= 0)
			{
				mesh.vertex_list.push_back(ldp::Float3(i, j,
					k + m_value[index(i, j, k)] / (m_value[index(i, j, k)] - m_value[index(i, j, k + 1)]))
					);
				z_vertices[index(i, j, k)] = mesh.vertex_list.size() - 1;
			}
		}

		FOR_EVERY_CELL if (i != m_size[0] - 1 && j != m_size[1] - 1 && k != m_size[2] - 1)
		{
			int CubeIndex = 0;
			if (m_value[index(i, j, k)]>0.0) CubeIndex |= 1;
			if (m_value[index(i + 1, j, k)]>0.0) CubeIndex |= 2;
			if (m_value[index(i + 1, j + 1, k)]>0.0) CubeIndex |= 4;
			if (m_value[index(i, j + 1, k)]>0.0) CubeIndex |= 8;
			if (m_value[index(i, j, k + 1)]>0.0) CubeIndex |= 16;
			if (m_value[index(i + 1, j, k + 1)]>0.0) CubeIndex |= 32;
			if (m_value[index(i + 1, j + 1, k + 1)]>0.0) CubeIndex |= 64;
			if (m_value[index(i, j + 1, k + 1)]>0.0) CubeIndex |= 128;

			int id;
			for (id = 0; triTable[CubeIndex][id] != -1; id++)
			{
				int ti = id % 3;
				if (ti == 0)
					mesh.face_list.push_back(ObjMesh::obj_face());
				ObjMesh::obj_face& f = mesh.face_list.back();
				f.vertex_count = 3;
				if (triTable[CubeIndex][id] == 0)	f.vertex_index[ti] = x_vertices[index(i, j, k)];
				if (triTable[CubeIndex][id] == 1)	f.vertex_index[ti] = y_vertices[index(i + 1, j, k)];
				if (triTable[CubeIndex][id] == 2)	f.vertex_index[ti] = x_vertices[index(i, j + 1, k)];
				if (triTable[CubeIndex][id] == 3)	f.vertex_index[ti] = y_vertices[index(i, j, k)];

				if (triTable[CubeIndex][id] == 4)	f.vertex_index[ti] = x_vertices[index(i, j, k + 1)];
				if (triTable[CubeIndex][id] == 5)	f.vertex_index[ti] = y_vertices[index(i + 1, j, k + 1)];
				if (triTable[CubeIndex][id] == 6)	f.vertex_index[ti] = x_vertices[index(i, j + 1, k + 1)];
				if (triTable[CubeIndex][id] == 7)	f.vertex_index[ti] = y_vertices[index(i, j, k + 1)];

				if (triTable[CubeIndex][id] == 8)	f.vertex_index[ti] = z_vertices[index(i, j, k)];
				if (triTable[CubeIndex][id] == 9)	f.vertex_index[ti] = z_vertices[index(i + 1, j, k)];
				if (triTable[CubeIndex][id] == 10)	f.vertex_index[ti] = z_vertices[index(i + 1, j + 1, k)];
				if (triTable[CubeIndex][id] == 11)	f.vertex_index[ti] = z_vertices[index(i, j + 1, k)];
			}
		}

		mesh.scaleBy(m_step, 0);
		mesh.translate(m_start);
	}

	void LevelSet3D::load(std::string filename)
	{
		std::fstream input(filename, std::ios::in | std::ios::binary);
		if (input.bad())
			throw std::exception(("IOError: " + filename).c_str());
		input.read((char*)m_size.ptr(), sizeof(m_size));
		input.read((char*)&m_step, sizeof(m_step));
		input.read((char*)m_start.ptr(), sizeof(m_start));

		create(m_size, m_start, m_step);

		int number = 0;
		input.read((char*)&number, sizeof(number));
		std::vector<int>index(number);
		std::vector<ValueType>value(number);
		input.read((char*)index.data(), number*sizeof(int));
		input.read((char*)value.data(), number*sizeof(ValueType));
#ifdef IMPORT_WANG_HUAMING_FORMAT
		std::vector<ValueType>u_value(number, 0);
		input.read((char*)u_value.data(), number*sizeof(ValueType));
		input.read((char*)u_value.data(), number*sizeof(ValueType));
		input.read((char*)u_value.data(), number*sizeof(ValueType));
#endif
		for (int i = 0; i < number; i++)
			m_value[index[i]] = value[i];
		input.close();
		printf("load data from file %s successfully.\n", filename.c_str());
		fastMarching();
	}

	void LevelSet3D::save(std::string filename)const
	{
		std::fstream output(filename, std::ios::out | std::ios::binary);
		if (output.fail())
			throw std::exception(("IOError: " + filename).c_str());
		output.write((const char*)m_size.ptr(), sizeof(m_size));
		output.write((const char*)&m_step, sizeof(m_step));
		output.write((const char*)m_start.ptr(), sizeof(m_start));

		int number = 0;
		for (int i = 0; i<sizeXYZ(); i++) 
		if (fabsf(m_value[i])<3) 
			number++;
		std::vector<int >index(number);
		std::vector<ValueType>value(number);
#ifdef EXPORT_WANG_HUAMING_FORMAT
		std::vector<ValueType>u_value(number, 0);
		std::vector<ValueType>v_value(number, 0);
		std::vector<ValueType>w_value(number, 0);
#endif
		int _number = 0;
		for (int i = 0; i<sizeXYZ(); i++)
		if (fabsf(m_value[i])<3)
		{
			index[_number] = i;
			value[_number] = m_value[i];
			_number++;
		}
		output.write((const char*)&number, sizeof(number));	
		output.write((const char*)index.data(), number*sizeof(int));
		output.write((const char*)value.data(), number*sizeof(ValueType));
#ifdef EXPORT_WANG_HUAMING_FORMAT
		output.write((const char*)u_value.data(), number*sizeof(ValueType));
		output.write((const char*)v_value.data(), number*sizeof(ValueType));
		output.write((const char*)w_value.data(), number*sizeof(ValueType));
#endif
		output.close();
		printf("Write data into file %s successfully.\n", filename.c_str());
	}

	void LevelSet3D::fastMarching(const int band_width, bool reinitialize, bool boundary)
	{
		std::vector<LEVEL_SET_CELL_DATA> c(m_value.size());
		FOR_EVERY_CELL
		{
			c[id].id = id;
			c[id].state = UNPROCESSED;
		}

		// Initialize the distance function	
		if (boundary == false)
		{
			FOR_EVERY_CELL
			{
				if (i<BUFFER || j<BUFFER || k<BUFFER || i>m_size[0] - BUFFER
				|| j>m_size[1] - BUFFER || k>m_size[2] - BUFFER)
					m_value[id] = 1;
				if (fabsf(m_value[id]) < EPSILON)
					m_value[id] = EPSILON*SIGN(m_value[id]);
			}
		}
		else
		{
			FOR_EVERY_CELL
			{
				int ii = i;
				int jj = j;
				int kk = k;
				if (i <= BUFFER)		ii = BUFFER + 1;
				if (j <= BUFFER)		jj = BUFFER + 1;
				if (k <= BUFFER)		kk = BUFFER + 1;
				if (i >= m_size[0] - BUFFER)	ii = m_size[0] - BUFFER - 1;
				if (j >= m_size[1] - BUFFER)	jj = m_size[1] - BUFFER - 1;
				if (k >= m_size[2] - BUFFER)	kk = m_size[2] - BUFFER - 1;
				m_value[id] = m_value[index(ii, jj, kk)];
				if (fabsf(m_value[id]) < EPSILON) 
					m_value[id] = EPSILON*SIGN(m_value[id]);
			}
		}

		// Reinitialize the distances of surface cells
		if (reinitialize == true)
		{
			std::vector<ValueType> temp = m_value;
			FOR_EVERY_CELL	if (m_value[id]<0)
			{
				if (i - 1 >= 0 && m_value[index(i - 1, j, k)] >= 0 && m_value[index(i - 1, j, k)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i - 1, j, k), temp, m_value, c);
				if (i + 1<m_size[0] && m_value[index(i + 1, j, k)] >= 0 && m_value[index(i + 1, j, k)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i + 1, j, k), temp, m_value, c);
				if (j - 1 >= 0 && m_value[index(i, j - 1, k)] >= 0 && m_value[index(i, j - 1, k)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i, j - 1, k), temp, m_value, c);
				if (j + 1<m_size[1] && m_value[index(i, j + 1, k)] >= 0 && m_value[index(i, j + 1, k)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i, j + 1, k), temp, m_value, c);
				if (k - 1 >= 0 && m_value[index(i, j, k - 1)] >= 0 && m_value[index(i, j, k - 1)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i, j, k - 1), temp, m_value, c);
				if (k + 1<m_size[2] && m_value[index(i, j, k + 1)] >= 0 && m_value[index(i, j, k + 1)] != MY_INFINITE)
					Process_Surface_Cell(id, index(i, j, k + 1), temp, m_value, c);
			}
			for (int i = 0; i<m_value.size(); i++)
			if (c[i].state == PROCESSED) 
				m_value[i] = temp[i];
		}

		// Initialize the heap
		HEAP<LEVEL_SET_CELL_DATA *> heap;
		FOR_EVERY_CELL if (c[id].state == PROCESSED)
		{
			if (i - 1 >= 0)		
				processCell(i - 1, j, k, heap, band_width, c);
			if (i + 1<m_size[0])	
				processCell(i + 1, j, k, heap, band_width, c);
			if (j - 1 >= 0)		
				processCell(i, j - 1, k, heap, band_width, c);
			if (j + 1<m_size[1])	
				processCell(i, j + 1, k, heap, band_width, c);
			if (k - 1 >= 0)		
				processCell(i, j, k - 1, heap, band_width, c);
			if (k + 1<m_size[2])	
				processCell(i, j, k + 1, heap, band_width, c);
		}

		// Run the process
		while (1)
		{
			LEVEL_SET_CELL_DATA *top_voxel = heap.Remove_Top();

			if (top_voxel == NULL) break;
			int i = top_voxel->id / sizeYZ();
			int j = (top_voxel->id%sizeYZ()) / m_size[2];
			int k = top_voxel->id%m_size[2];

			if (c[index(i, j, k)].state != PROCESSING) 
			{ 
				printf("ERROR: cells are not in processing (Fast Marching).\n"); 
				getchar(); 
			}
			c[index(i, j, k)].state = PROCESSED;

			if (m_value[index(i, j, k)] >= band_width)	
				continue;
			if (i - 1 >= 0)		
				processCell(i - 1, j, k, heap, band_width, c);
			if (i + 1<m_size[0])	
				processCell(i + 1, j, k, heap, band_width, c);
			if (j - 1 >= 0)		
				processCell(i, j - 1, k, heap, band_width, c);
			if (j + 1<m_size[1])	
				processCell(i, j + 1, k, heap, band_width, c);
			if (k - 1 >= 0)		
				processCell(i, j, k - 1, heap, band_width, c);
			if (k + 1<m_size[2])	
				processCell(i, j, k + 1, heap, band_width, c);
		}

		// Update the remaining cells
		FOR_EVERY_CELL	if (c[id].state != PROCESSED)
		{
			if (m_value[id]<0) 
			{ 
				printf("ERROR: %d, %d, %d is not processed (Fast Marching).\n", i, j, k);	
				getchar(); 
			}
			m_value[id] = MY_INFINITE*SIGN(m_value[id]);
		}
	}

	void LevelSet3D::processCell(int i, int j, int k, HEAP <LEVEL_SET_CELL_DATA *> &heap_tree, 
		ValueType band_width, std::vector<LEVEL_SET_CELL_DATA>& c)
	{
		int id = index(i, j, k);
		if (c[id].state == PROCESSED) return;
		if (c[id].state == UNPROCESSED)
		{
			ValueType sign;
			computeExtendedDistance(i, j, k, sign, c);
			if (m_value[id]<band_width)  //valid BUFFER.
			{
				heap_tree.Add(&c[id], m_value[id] * sign);
				c[id].state = PROCESSING;
			}
		}
		else if (c[id].state == PROCESSING)
		{
			ValueType sign;
			computeExtendedDistance(i, j, k, sign, c);
			heap_tree.Update_Key(c[id].heap_node, m_value[id] * sign);
		}
		else 
		{ 
			printf("ERROR: Unknown state %d.\n", c[index(i, j, k)].state); 
			getchar();
		}
	}

	LevelSet3D::ValueType LevelSet3D::computeExtendedDistance(int i, int j, int k, ValueType &sign,
		std::vector<LEVEL_SET_CELL_DATA>& c)
	{
		ValueType t_sign;
		ValueType theta1 = MY_INFINITE, theta2 = MY_INFINITE, theta3 = MY_INFINITE;
		sign = 0;//Sign(*c[INDEX(i,j,k)].p);

		// X direction
		if (i - 1 >= 0 && c[index(i - 1, j, k)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i - 1, j, k)]);
			if (sign && sign != t_sign) 
			{ 
				printf("Error. not consistent in Compute_Extended_Distance.1, %d, %d, %d (%d, %f)\n", 
					i, j, k, t_sign, m_value[index(i, j + 1, k)]); 
				while (1); 
			}
			sign = t_sign;
			theta1 = m_value[index(i - 1, j, k)];
		}
		if (i + 1<m_size[0] && c[index(i + 1, j, k)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i + 1, j, k)]);
			if (sign && sign != t_sign) 
			{ 
				printf("Error. not consistent in Compute_Extended_Distance.2, %d, %d, %d (%d, %f)\n", 
					i, j, k, t_sign, m_value[index(i, j + 1, k)]);
				while (1); 
			}
			else sign = t_sign;
			if (theta1 == MY_INFINITE || fabsf(m_value[index(i + 1, j, k)])<fabsf(theta1))
				theta1 = m_value[index(i + 1, j, k)];
		}
		//Y direction
		if (j - 1 >= 0 && c[index(i, j - 1, k)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i, j - 1, k)]);
			if (sign && sign != t_sign) 
			{
				printf("Error. not consistent in sign when Compute_Extended_Distance.3, %d, %d, %d\n", i, j, k); 
				while (1); 
			}
			else sign = t_sign;
			theta2 = m_value[index(i, j - 1, k)];
		}
		if (j + 1<m_size[1] && c[index(i, j + 1, k)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i, j + 1, k)]);
			if (sign && sign != t_sign) 
			{ 
				printf("Error. not consistent in Compute_Extended_Distance.4, %d, %d, %d (%d, %f)\n", 
					i, j, k, t_sign, m_value[index(i, j + 1, k)]);
				while (1); 
			}
			else sign = t_sign;
			if (theta2 == MY_INFINITE || fabsf(m_value[index(i, j + 1, k)])<fabsf(theta2))
				theta2 = m_value[index(i, j + 1, k)];
		}
		//Z direction
		if (k - 1 >= 0 && c[index(i, j, k - 1)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i, j, k - 1)]);
			if (sign && sign != t_sign) 
			{ 
				printf("Error. not consistent in sign when Compute_Extended_Distance.5\n"); 
				while (1); 
			}
			else sign = t_sign;
			theta3 = m_value[index(i, j, k - 1)];
		}
		if (k + 1<m_size[2] && c[index(i, j, k + 1)].state == PROCESSED)
		{
			t_sign = SIGN(m_value[index(i, j, k + 1)]);
			if (sign && sign != t_sign) 
			{ 
				printf("here: %f; %f\n", sign, t_sign); 
				printf("Error. not consistent in sign when Compute_Extended_Distance.6\n"); 
				while (1); 
			}
			else sign = t_sign;
			if (theta3 == MY_INFINITE || fabsf(m_value[index(i, j, k + 1)])<fabsf(theta3))
				theta3 = m_value[index(i, j, k + 1)];
		}
		return m_value[index(i, j, k)] = Extended_Distance_With_Three_Value(theta1, theta2, theta3, sign);
	}
}