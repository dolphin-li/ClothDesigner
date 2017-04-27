#pragma once

#include <eigen\Dense>
#include <eigen\Sparse>
#include <memory>
class ObjMesh;
namespace ldp
{
	class LoopSubdiv
	{
	public:
		typedef Eigen::SparseMatrix<float> SpMat;
		typedef Eigen::VectorXf Vec;
		typedef Eigen::MatrixXf Mat;
	public:
		LoopSubdiv();
		~LoopSubdiv();

		void clear();
		void init(ObjMesh* objMesh);

		int numInputVerts()const{ return m_subdivMat.cols(); }
		void updateTopology();
		void run();

		const ObjMesh* getResultMesh()const{ return m_resultMesh.get(); }
		ObjMesh* getResultMesh(){ return m_resultMesh.get(); }
	private:
		ObjMesh* m_inputMesh = nullptr;
		SpMat m_subdivMat;
		Mat m_inputVerts;
		Mat m_outputVerts;
		std::shared_ptr<ObjMesh> m_resultMesh;
	};
}