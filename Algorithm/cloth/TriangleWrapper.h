#pragma once

#include "definations.h"
#include <hash_map>
#include <map>
extern "C"{
	struct triangulateio;
}
namespace ldp
{
	class ClothPiece;
	class Sewing;
	class PanelPolygon;
	class ShapeGroup;
	class AbstractShape;
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

		const std::vector<StitchPointPair>& sewingVertPairs()const { return m_stitches; }
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
		struct SampleParam
		{
			float t;
			int idx;
			SampleParam(float tt, int ii) :t(tt), idx(ii) {}
			SampleParam() :t(0), idx(0) {}
		};
		struct SampleParamVec
		{
			std::vector<SampleParam> params;
			const AbstractShape* shape;
			float step;
			float start;
			float end;
			SampleParamVec() :step(0), start(0), end(0), shape(nullptr) {}
		};
		typedef std::shared_ptr<SampleParamVec> SampleParamVecPtr;
		typedef	std::vector<SampleParamVecPtr> ShapeSegs;
		typedef std::shared_ptr<ShapeSegs> ShapeSegsPtr;
		void createShapeSeg(const AbstractShape* shape, float step);
		int addSegToShape(ShapeSegs& segs, float tSegs);
		void resampleSeg(SampleParamVec& segs, float step);
		struct SegPair
		{
			SampleParamVec* seg[2];
			bool reverse[2];
			SegPair(SampleParamVec* seg1, bool reverse1,
				SampleParamVec* seg2, bool reverse2)
			{
				seg[0] = seg1;
				reverse[0] = reverse1;
				seg[1] = seg2;
				reverse[1] = reverse2;
			}
		};
		typedef std::vector<SegPair> SegPairVec;
		void updateSegStepMap(SampleParamVec* seg, float step);
	private:
		/// computing structure
		triangulateio* m_in = nullptr;
		triangulateio* m_out = nullptr;
		triangulateio* m_vro = nullptr;
		char m_cmds[1024];

		/// input
		std::vector<std::shared_ptr<ClothPiece>>* m_pieces = nullptr;
		std::vector<std::shared_ptr<Sewing>>* m_sewings = nullptr;
		float m_ptMergeThre = 0;
		float m_triSize = 0;
		float m_ptOnLineThre = 0;

		/// intermediate input buffer for triangle
		std::vector<Double2> m_points;
		std::vector<Int2> m_segments;
		std::vector<Double2> m_holeCenters;

		/// intermediate output buffer of triangle
		std::vector<Double2> m_triVertsBuffer;
		std::vector<Int3> m_triBuffer;

		/// sewing related
		std::vector<StitchPointPair> m_stitches;;
		std::hash_map<const AbstractShape*, const ClothPiece*> m_shapePieceMap;
		std::hash_map<const ClothPiece*, int> m_vertStart;
		std::hash_map<const AbstractShape*, ShapeSegsPtr> m_shapeSegs;
		SegPairVec m_segPairs;
		std::map<SampleParamVec*, float> m_segStepMap;
	};
}