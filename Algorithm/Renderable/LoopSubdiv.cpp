#include "LoopSubdiv.h"
#include "ObjMesh.h"
namespace ldp
{
	LoopSubdiv::LoopSubdiv()
	{
		m_resultMesh.reset(new ObjMesh);
	}

	LoopSubdiv::~LoopSubdiv()
	{
		clear();
	}

	void LoopSubdiv::clear()
	{
		m_resultMesh->clear();
		m_inputMesh = nullptr;
		m_inputVerts.resize(0, 0);
		m_subdivMat.resize(0, 0);
		m_outputVerts.resize(0, 0);
	}

	void LoopSubdiv::init(ObjMesh* objMesh)
	{
		clear();
		for (const auto& f : objMesh->face_list)
		if (f.vertex_count != 3)
			throw std::exception("LoopSubdiv: only triangle mesh supported!");

		m_inputMesh = objMesh;
		updateTopology();
	}

	void LoopSubdiv::updateTopology()
	{
		m_resultMesh->clear();
		m_resultMesh->material_list = m_inputMesh->material_list;
		BMesh& bmesh = *m_inputMesh->get_bmesh(false);
		const int nVerts = m_inputMesh->vertex_list.size();
		const int nEdges = bmesh.eofm_count();
		const int nFaces = m_inputMesh->face_list.size();
		m_inputVerts.resize(nVerts, 3);
		m_outputVerts.resize(nVerts + nEdges, 3);

		// construct face topology
		BMESH_ALL_FACES(f, f_of_m_iter, bmesh)
		{
			BMEdge* edges[3] = { 0 };
			BMVert* verts[3] = { 0 };
			int cnt = 0;
			BMESH_E_OF_F(e, f, e_of_f_iter, bmesh)
			{
				edges[cnt++] = e;
			}
			verts[0] = bmesh.vofe_first(edges[0]);
			verts[1] = bmesh.vofe_last(edges[0]);
			if (verts[0] != bmesh.vofe_first(edges[2]) && verts[0] != bmesh.vofe_last(edges[2]))
				std::swap(verts[0], verts[1]);
			verts[2] = bmesh.vofe_first(edges[2]) == verts[0] ? bmesh.vofe_last(edges[2]) : bmesh.vofe_first(edges[2]);

			const ObjMesh::obj_face& oriFace = m_inputMesh->face_list[f->getIndex()];
			ObjMesh::obj_face f = oriFace;
			f.material_index = std::min(f.material_index, (int)m_resultMesh->material_list.size() - 1);
			f.vertex_index[0] = verts[0]->getIndex();
			f.vertex_index[1] = edges[0]->getIndex() + nVerts;
			f.vertex_index[2] = edges[2]->getIndex() + nVerts;
			m_resultMesh->face_list.push_back(f);
			f.vertex_index[0] = verts[1]->getIndex();
			f.vertex_index[1] = edges[1]->getIndex() + nVerts;
			f.vertex_index[2] = edges[0]->getIndex() + nVerts;
			m_resultMesh->face_list.push_back(f);
			f.vertex_index[0] = verts[2]->getIndex();
			f.vertex_index[1] = edges[2]->getIndex() + nVerts;
			f.vertex_index[2] = edges[1]->getIndex() + nVerts;
			m_resultMesh->face_list.push_back(f);
			f.vertex_index[0] = edges[0]->getIndex() + nVerts;
			f.vertex_index[1] = edges[1]->getIndex() + nVerts;
			f.vertex_index[2] = edges[2]->getIndex() + nVerts;
			m_resultMesh->face_list.push_back(f);
		} // end for all faces

		std::vector<Eigen::Triplet<float>> cooSys;

		// compute edge verts
		std::vector<int> ids;
		BMESH_ALL_EDGES(e, e_of_m_iter, bmesh)
		{
			ids.clear();
			ids.push_back(bmesh.vofe_first(e)->getIndex());
			ids.push_back(bmesh.vofe_last(e)->getIndex());
			BMESH_F_OF_E(f, e, f_of_e_iter, bmesh)
			{
				BMESH_V_OF_F(v, f, v_of_f_iter, bmesh)
				{
					if (v->getIndex() != ids[0] && v->getIndex() != ids[1])
					{
						ids.push_back(v->getIndex());
						break;
					}
				}
			}
			assert(ids.size() >= 1);
			const int row = e->getIndex() + nVerts;
			if (ids.size() != 4)
			{
				cooSys.push_back(Eigen::Triplet<float>(row, ids[0], 0.5f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[1], 0.5f));
			}
			else
			{
				cooSys.push_back(Eigen::Triplet<float>(row, ids[0], 0.375f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[1], 0.375f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[2], 0.125f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[3], 0.125f));
			}
		} // end for edges

		// compute vert verts
		std::vector<int> boundary_id;
		BMESH_ALL_VERTS(v, v_of_m_iter, bmesh)
		{
			ids.clear();
			boundary_id.clear();
			BMESH_E_OF_V(e, v, e_of_v_iter, bmesh)
			{
				BMVert* v1 = bmesh.vofe_first(e) == v ? bmesh.vofe_last(e) : bmesh.vofe_first(e);
				if (bmesh.fofe_count(e) != 2)
					boundary_id.push_back(ids.size());
				ids.push_back(v1->getIndex());
			}
			const int row = v->getIndex();
			if (boundary_id.size() == 2)
			{
				cooSys.push_back(Eigen::Triplet<float>(row, row, 0.75f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[boundary_id[0]], 0.125f));
				cooSys.push_back(Eigen::Triplet<float>(row, ids[boundary_id[1]], 0.125f));
			}
			else
			{
				const float beta = ids.size() == 3 ? 3.f / 16.f : 3.f / (8.f*ids.size());
				cooSys.push_back(Eigen::Triplet<float>(row, row, 1.f - beta * ids.size()));
				for (const auto& id : ids)
					cooSys.push_back(Eigen::Triplet<float>(row, id, beta));
			}
		} // end for verts

		m_resultMesh->vertex_list.resize(nVerts + nEdges);
		m_subdivMat.resize(nVerts + nEdges, nVerts);
		if (!cooSys.empty())
			m_subdivMat.setFromTriplets(cooSys.begin(), cooSys.end());

		// perform numerical subdiv
		run();
	}

	void LoopSubdiv::run()
	{
		if (m_inputMesh == nullptr)
		{
			printf("LoopSubdiv::run(), warning: not initailzed!\n");
			return;
		}

		if (m_subdivMat.cols() != m_inputMesh->vertex_list.size())
			updateTopology();

		for (size_t iVert = 0; iVert < m_inputMesh->vertex_list.size(); iVert++)
		for (int k = 0; k < 3; k++)
			m_inputVerts(iVert, k) = m_inputMesh->vertex_list[iVert][k];

		m_outputVerts = m_subdivMat * m_inputVerts;

		for (size_t iVert = 0; iVert < m_resultMesh->vertex_list.size(); iVert++)
		for (int k = 0; k < 3; k++)
			m_resultMesh->vertex_list[iVert][k] = m_outputVerts(iVert, k);


		for (size_t iFace = 0; iFace < m_inputMesh->face_list.size(); iFace++)
		{
			const int mat = m_inputMesh->face_list[iFace].material_index;
			for (int k = 0; k < 4; k++)
				m_resultMesh->face_list[iFace * 4 + k].material_index = mat;
		}

		m_resultMesh->updateNormals();
		m_resultMesh->updateBoundingBox();
	}
}