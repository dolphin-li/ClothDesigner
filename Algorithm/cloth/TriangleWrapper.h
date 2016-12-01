#pragma once

#include "ldpMat\ldp_basic_vec.h"
extern "C"{
	struct triangulateio;
}
namespace ldp
{
	class ClothPiece;
	class Sewing;
	class PanelPolygon;
	class ShapeGroup;
	class TriangleWrapper
	{
	public:
		TriangleWrapper();
		~TriangleWrapper();

		void triangulate(
			std::vector<std::shared_ptr<ClothPiece>>& pieces, 
			std::vector<std::shared_ptr<Sewing>>& sewings,
			float pointMergeThre,
			float triangleSize,
			float pointOnLineThre
			);

		const std::vector<Int2>& sewingVertPairs()const { return m_sewingVertPairs; }
	protected:
		void reset_triangle_struct(triangulateio* io)const;
		void prepareTriangulation();
		void precomputeSewing();
		void addPolygon(const ShapeGroup& poly);
		void addDart(const ShapeGroup& dart);
		void addLine(const ShapeGroup& line);
		void finalizeTriangulation();
		void generateMesh(ClothPiece& piece);
		void postComputeSewing();
	private:
		/// computing structure
		triangulateio* m_in;
		triangulateio* m_out;
		triangulateio* m_vro;
		char m_cmds[1024];

		/// input
		std::vector<std::shared_ptr<ClothPiece>>* m_pieces;
		std::vector<std::shared_ptr<Sewing>>* m_sewings;
		float m_ptMergeThre;
		float m_triSize;
		float m_ptOnLineThre;

		/// intermediate input buffer for triangle
		std::vector<Double2> m_points;
		std::vector<Int2> m_segments;
		std::vector<Double2> m_holeCenters;

		/// intermediate output buffer of triangle
		std::vector<Double2> m_triVertsBuffer;
		std::vector<Int3> m_triBuffer;

		/// sewing related
		std::vector<Int2> m_sewingVertPairs;
	};
}