#include <assert.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
#include "BGraph.h"
#include "GraphLine.h"
#include "cloth\definations.h"
using namespace std;

#ifndef NULL
#define NULL 0
#endif
namespace ldp
{
	typedef struct BGAllocTemplate
	{
		int totvert, totedge, totloop, totface;
	} BGAllocTemplate;
	BGAllocTemplate bm_mesh_chunksize_default = { 512, 1024, 2048, 512 };
	/************************************************************************/
	/* Small math functions /structures needed
	/************************************************************************/
#include "bgraph_private.h"

	/************************************************************************/
	/* BGIter
	/************************************************************************/
	void BGIter::init(BGLoop* l)
	{
		clear();
		ldata = l;
	}
	void BGIter::init(const BGVert* v)
	{
		clear();
		c_vdata = v;
	}
	void BGIter::init(const BGFace* f)
	{
		clear();
		c_pdata = f;
	}
	void BGIter::init(const BGEdge* e)
	{
		clear();
		c_edata = e;
	}
	void BGIter::clear()
	{
		firstvert = nextvert = 0;
		firstedge = nextedge = 0;
		firstloop = nextloop = ldata = l = 0;
		firstpoly = nextpoly = 0;
		c_vdata = 0;
		c_edata = 0;
		c_pdata = 0;
	}
	/************************************************************************/
	/* BGESH
	/************************************************************************/
	BGraph::BGraph() :AbstractGraphObject()
	{

	}

	BGraph::~BGraph()
	{

	}

	BGraph* BGraph::clone()const
	{
		const BGraph* bm_old = this;
		BGraph* bm_new = new BGraph();
		m_objMapAfterClone.clear();
		std::hash_map<const void*, void*> objMap;
		BGRAPH_ALL_VERTS_CONST(v, viter, *bm_old)
		{
			auto newObj = bm_new->create_vert(v->position());
			objMap[v] = newObj;
			m_objMapAfterClone[(BGVert*)v] = newObj;
		}
		BGRAPH_ALL_EDGES_CONST(e, eiter, *bm_old)
		{
			std::vector<BGVert*> vptrs;
			for (auto& p : e->m_keyPoints)
				vptrs.push_back((BGVert*)objMap[p]);
			auto newObj = bm_new->create_edge(vptrs);
			m_objMapAfterClone[(AbstractGraphObject*)e] = newObj;
		}
		BGRAPH_ALL_FACES_CONST(f, fiter, *bm_old)
		{
			std::vector<BGVert*> oldVerts, newVerts;
			std::vector<BGEdge*> edges;

			BGRAPH_V_OF_F(v, f, viter, *bm_old)
			{
				oldVerts.push_back(v);
				newVerts.push_back((BGVert*)objMap[v]);
			}
			for (int j = 0; j < f->len; j++)
			{
				BGEdge* e = bm_old->eofv_2(oldVerts[j], oldVerts[(j + 1) % f->len]);
				assert(e);
				edges.push_back((BGEdge*)objMap[e]);
			}

			auto newObj = bm_new->create_face(newVerts.data(), edges.data(), f->len);
			objMap[f] = newObj;
			m_objMapAfterClone[(AbstractGraphObject*)f] = newObj;
		}
		return bm_new;
	}


	TiXmlElement* BGraph::toXML(TiXmlNode* parent)const
	{
		return nullptr;
	}

	void BGraph::fromXML(TiXmlElement* self)
	{

	}

	/************************************************************************/
	/* High Level functions
	/************************************************************************/
	/**
	* Queries: Vert Of Mesh.
	* */
	BGVert* BGraph::vofm_begin(BGIter& iter)
	{
		if (vpool.empty())
			return nullptr;
		iter.viter = vpool.begin();
		return iter.viter->second.get();
	}
	BGVert* BGraph::vofm_next(BGIter& iter)
	{
		++iter.viter;
		if (iter.viter == vpool.end())
			return nullptr;
		return iter.viter->second.get();
	}
	BGVert* BGraph::vofm_end(BGIter& iter)
	{
		return 0;
	}
	const BGVert* BGraph::vofm_begin(BGIter& iter)const
	{
		if (vpool.empty())
			return nullptr;
		iter.cviter = vpool.begin();
		return iter.cviter->second.get();
	}
	const BGVert* BGraph::vofm_next(BGIter& iter)const
	{
		++iter.cviter;
		if (iter.cviter == vpool.end())
			return nullptr;
		return iter.cviter->second.get();
	}
	const BGVert* BGraph::vofm_end(BGIter& iter)const
	{
		++iter.cviter;
		if (iter.cviter == vpool.end())
			return nullptr;
		return iter.cviter->second.get();
	}
	/**
	* Queries: Edge Of Mesh.
	* */
	BGEdge* BGraph::eofm_begin(BGIter& iter)
	{
		if (epool.empty())
			return nullptr;
		iter.eiter = epool.begin();
		return iter.eiter->second.get();
	}
	BGEdge* BGraph::eofm_next(BGIter& iter)
	{
		++iter.eiter;
		if (iter.eiter == epool.end())
			return nullptr;
		return iter.eiter->second.get();
	}
	BGEdge* BGraph::eofm_end(BGIter& iter)
	{
		return 0;
	}
	const BGEdge* BGraph::eofm_begin(BGIter& iter)const
	{
		if (epool.empty())
			return nullptr;
		iter.ceiter = epool.begin();
		return iter.ceiter->second.get();
	}
	const BGEdge* BGraph::eofm_next(BGIter& iter)const
	{
		++iter.ceiter;
		if (iter.ceiter == epool.end())
			return nullptr;
		return iter.ceiter->second.get();
	}
	const BGEdge* BGraph::eofm_end(BGIter& iter)const
	{
		return 0;
	}
	/**
	* Queries: Face Of Mesh
	* */
	BGFace* BGraph::fofm_begin(BGIter& iter)
	{
		if (fpool.empty())
			return nullptr;
		iter.fiter = fpool.begin();
		return iter.fiter->second.get();
	}
	BGFace* BGraph::fofm_next(BGIter& iter)
	{
		++iter.fiter;
		if (iter.fiter == fpool.end())
			return nullptr;
		return iter.fiter->second.get();
	}
	BGFace* BGraph::fofm_end(BGIter& iter)
	{
		return 0;
	}
	const BGFace* BGraph::fofm_begin(BGIter& iter)const
	{
		if (fpool.empty())
			return nullptr;
		iter.cfiter = fpool.begin();
		return iter.cfiter->second.get();
	}
	const BGFace* BGraph::fofm_next(BGIter& iter)const
	{
		++iter.cfiter;
		if (iter.cfiter == fpool.end())
			return nullptr;
		return iter.cfiter->second.get();
	}
	const BGFace* BGraph::fofm_end(BGIter& iter)const
	{
		return 0;
	}


	//vert of face
	BGVert* BGraph::voff_begin(BGIter& iter)
	{
		assert(iter.c_pdata);
		iter.l = iter.c_pdata->l_first;
		return iter.l ? iter.l->v : 0;
	}
	BGVert* BGraph::voff_next(BGIter& iter)
	{
		if (iter.l)	iter.l = iter.l->next;
		if (iter.l == iter.c_pdata->l_first) iter.l = 0;
		return iter.l ? iter.l->v : 0;
	}
	BGVert* BGraph::voff_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::voff_count(const BGFace* f)
	{
		return f->len;
	}

	//edge of vert
	BGEdge* BGraph::eofv_begin(BGIter& iter)
	{
		assert(iter.c_vdata);
		iter.nextedge = iter.c_vdata->m_edge;
		return iter.c_vdata->m_edge;
	}
	BGEdge* BGraph::eofv_next(BGIter& iter)
	{
		if (iter.nextedge == 0)	return 0;
		BGEdge* e = bmesh_disk_edge_next(iter.nextedge, iter.c_vdata);
		if (e == iter.c_vdata->m_edge)	e = 0;
		iter.nextedge = e;
		return e;
	}
	BGEdge* BGraph::eofv_end(BGIter& iter)
	{
		return 0;
	}
	BGEdge* BGraph::eofv_2(BGVert* v1, BGVert* v2)
	{
		BGIter iter;
		iter.init(v1);
		for (BGEdge* e = eofv_begin(iter); e != eofv_end(iter); e = eofv_next(iter))
		{
			if (e->getStartPoint() == v2 || e->getEndPoint() == v2)
				return e;
		}
		return 0;
	}
	int BGraph::eofv_count(BGVert* v)
	{
		BGIter iter;
		int count = 0;
		iter.init(v);
		for (BGEdge* e = eofv_begin(iter); e != eofv_end(iter); e = eofv_next(iter))
		{
			count++;
		}
		return count;
	}

	//edge of face
	BGEdge* BGraph::eoff_begin(BGIter& iter)
	{
		iter.l = iter.c_pdata->l_first;
		return iter.l ? iter.l->e : 0;
	}
	BGEdge* BGraph::eoff_next(BGIter& iter)
	{
		if (iter.l)	iter.l = iter.l->next;
		if (iter.l == iter.c_pdata->l_first) iter.l = 0;
		return iter.l ? iter.l->e : 0;
	}
	BGEdge* BGraph::eoff_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::eoff_count(BGFace* f)
	{
		BGIter iter;
		int count = 0;
		iter.init(f);
		for (BGEdge* e = eoff_begin(iter); e != eoff_end(iter); e = eoff_next(iter))
		{
			count++;
		}
		return count;
	}

	//face of vert
	BGFace* BGraph::fofv_begin(BGIter& iter)
	{
		assert(iter.c_vdata);
		iter.count = 0;
		if (iter.c_vdata->m_edge) iter.count = bmesh_disk_facevert_count(iter.c_vdata);
		if (iter.count)
		{
			iter.firstedge = bmesh_disk_faceedge_find_first(iter.c_vdata->m_edge, iter.c_vdata);
			iter.nextedge = iter.firstedge;
			iter.firstloop = bmesh_radial_faceloop_find_first(iter.firstedge->l, iter.c_vdata);
			iter.nextloop = iter.firstloop;
			return iter.firstloop ? iter.firstloop->f : 0;
		}
		return 0;
	}
	BGFace* BGraph::fofv_next(BGIter& iter)
	{
		BGLoop* current = iter.nextloop;
		if (iter.count && iter.nextloop)
		{
			iter.count--;
			iter.nextloop = bmesh_radial_faceloop_find_next(iter.nextloop, iter.c_vdata);
			if (iter.nextloop == iter.firstloop)
			{
				iter.nextedge = bmesh_disk_faceedge_find_next(iter.nextedge, iter.c_vdata);
				iter.firstloop = bmesh_radial_faceloop_find_first(iter.nextedge->l, iter.c_vdata);
				iter.nextloop = iter.firstloop;
			}
		}

		if (!iter.count) iter.nextloop = 0;
		return iter.nextloop ? iter.nextloop->f : 0;
	}
	BGFace* BGraph::fofv_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::fofv_count(BGVert* v)
	{
		BGIter iter;
		int count = 0;
		iter.init(v);
		for (BGFace* f = fofv_begin(iter); f != fofv_end(iter); f = fofv_next(iter))
		{
			count++;
		}
		return count;
	}

	//face of edge
	BGFace* BGraph::fofe_begin(BGIter& iter)
	{
		assert(iter.c_edata);
		if (iter.c_edata->l)
		{
			iter.firstloop = iter.nextloop = iter.c_edata->l;
			return iter.nextloop->f;
		}
		return 0;
	}
	BGFace* BGraph::fofe_next(BGIter& iter)
	{
		if (iter.nextloop)
			iter.nextloop = iter.nextloop->radial_next;
		if (iter.nextloop == iter.firstloop)	iter.nextloop = 0;
		return iter.nextloop ? iter.nextloop->f : 0;
	}
	BGFace* BGraph::fofe_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::fofe_count(BGEdge* e)
	{
		BGIter iter;
		int count = 0;
		iter.init(e);
		for (BGFace* f = fofe_begin(iter); f != fofe_end(iter); f = fofe_next(iter))
		{
			count++;
		}
		return count;
	}

	/**
	* Queries: Loop Of Vert
	* Useage: BGIter iter; iter.init(v); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	BGLoop* BGraph::lofv_begin(BGIter& iter)
	{
		assert(iter.c_vdata);
		iter.count = 0;
		if (iter.c_vdata->m_edge)
			iter.count = bmesh_disk_facevert_count(iter.c_vdata);
		if (iter.count)
		{
			iter.firstedge = bmesh_disk_faceedge_find_first(iter.c_vdata->m_edge, iter.c_vdata);
			iter.nextedge = iter.firstedge;
			iter.firstloop = bmesh_radial_faceloop_find_first(iter.firstedge->l, iter.c_vdata);
			iter.nextloop = iter.firstloop;
			return iter.nextloop;
		}
		return 0;
	}
	BGLoop* BGraph::lofv_next(BGIter& iter)
	{
		if (iter.count)
		{
			iter.count--;
			iter.nextloop = bmesh_radial_faceloop_find_next(iter.nextloop, iter.c_vdata);
			if (iter.nextloop == iter.firstloop)
			{
				iter.nextedge = bmesh_disk_faceedge_find_next(iter.nextedge, iter.c_vdata);
				iter.firstloop = bmesh_radial_faceloop_find_first(iter.nextedge->l, iter.c_vdata);
				iter.nextloop = iter.firstloop;
			}
		}

		if (!iter.count) iter.nextloop = NULL;
		return iter.nextloop;
	}
	BGLoop* BGraph::lofv_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::lofv_count(BGVert* v)
	{
		BGIter iter;
		int count = 0;
		iter.init(v);
		for (BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l = lofv_next(iter))
		{
			count++;
		}
		return count;
	}
	/**
	* Queries: Loop Of Edge
	* Useage: BGIter iter; iter.init(e); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	BGLoop* BGraph::lofe_begin(BGIter& iter)
	{
		assert(iter.c_edata);
		BGLoop *l;
		l = iter.c_edata->l;
		iter.firstloop = iter.nextloop = l;
		return l;
	}
	BGLoop* BGraph::lofe_next(BGIter& iter)
	{
		if (iter.nextloop)
			iter.nextloop = iter.nextloop->radial_next;
		if (iter.nextloop == iter.firstloop)
			iter.nextloop = 0;
		return iter.nextloop;
	}
	BGLoop* BGraph::lofe_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::lofe_count(BGEdge* e)
	{
		BGIter iter;
		int count = 0;
		iter.init(e);
		for (BGLoop* l = lofe_begin(iter); l != lofe_end(iter); l = lofe_next(iter))
		{
			count++;
		}
		return count;
	}
	/**
	* Queries: Loop Of Face
	* Useage: BGIter iter; iter.init(f); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
	* */
	BGLoop* BGraph::loff_begin(BGIter& iter)
	{
		iter.firstloop = iter.nextloop = iter.c_pdata->l_first;
		return iter.nextloop;
	}
	BGLoop* BGraph::loff_next(BGIter& iter)
	{
		if (iter.nextloop) iter.nextloop = iter.nextloop->next;
		if (iter.nextloop == iter.firstloop) iter.nextloop = 0;
		return iter.nextloop;
	}
	BGLoop* BGraph::loff_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::loff_count(BGFace* f)
	{
		BGIter iter;
		int count = 0;
		iter.init(f);
		for (BGLoop* l = loff_begin(iter); l != loff_end(iter); l = loff_next(iter))
		{
			count++;
		}
		return count;
	}

	//loops of loop
	BGLoop* BGraph::lofl_begin(BGIter& iter)
	{
		BGLoop *l;

		l = iter.ldata;
		iter.firstloop = l;
		iter.nextloop = iter.firstloop->radial_next;

		if (iter.nextloop == iter.firstloop)
			iter.nextloop = 0;
		return iter.nextloop;
	}
	BGLoop* BGraph::lofl_next(BGIter& iter)
	{
		if (iter.nextloop)
			iter.nextloop = iter.nextloop->radial_next;
		if (iter.nextloop == iter.firstloop)
			iter.nextloop = 0;
		return iter.nextloop;
	}
	BGLoop* BGraph::lofl_end(BGIter& iter)
	{
		return 0;
	}
	int BGraph::lofl_count(BGLoop* a_l)
	{
		BGIter iter;
		int count = 0;
		iter.init(a_l);
		for (BGLoop* l = lofl_begin(iter); l != lofl_end(iter); l = lofl_next(iter))
		{
			count++;
		}
		return count;
	}

	/************************************************************************/
	/* Low Level functions
	/************************************************************************/
	BGVert* BGraph::create_vert(Float2 p)
	{
		std::shared_ptr<BGVert> v(new BGVert);
		v->position() = p;
		return create_vert(v);
	}

	BGVert* BGraph::create_vert(const std::shared_ptr<BGVert>& rhs)
	{
		auto iter = vpool.find(rhs->getId());
		if (iter != vpool.end())
			return iter->second.get();
		for (auto iter : vpool)
		{
			if ((iter.second->position() - rhs->position()).length() < g_designParam.pointMergeDistThre)
				return iter.second.get();
		}
		vpool.insert(std::make_pair(rhs->getId(), rhs));
		return rhs.get();
	}

	BGFace*	BGraph::create_face(const std::vector<std::vector<std::shared_ptr<BGVert>>>& edges)
	{
		std::vector<BGEdge*> eptrs;
		for (auto& e : edges)
			eptrs.push_back(create_edge(e));
		return create_face(eptrs);
	}

	BGFace*	BGraph::create_face(const std::vector<std::shared_ptr<BGEdge>>& edges)
	{
		std::vector<BGEdge*> eptrs;
		for (auto& e : edges)
			eptrs.push_back(create_edge(e));
		return create_face(eptrs);
	}

	BGFace*	BGraph::create_face(const std::vector<BGEdge*>& curves)
	{
		if (curves.size() == 0)
			return nullptr;

		for (auto& c : curves)
		{
			if (epool.find(c->getId()) == epool.end())
				throw std::exception("create_face: curve not exist!");
		}

		// check the connectivity
		std::vector<int> shouldCurveReverse(curves.size(), 0);
		for (size_t i = 1; i < curves.size(); i++)
		{
			if (i == 1)
			{
				if (curves[0]->getEndPoint() != curves[1]->getStartPoint()
					&& curves[0]->getEndPoint() != curves[1]->getEndPoint())
					shouldCurveReverse[0] = 1;
			}
			auto lp = shouldCurveReverse[i - 1] ? curves[i - 1]->getStartPoint() : curves[i - 1]->getEndPoint();
			if (curves[i]->getStartPoint() == lp)
				shouldCurveReverse[i] = 0;
			else if (curves[i]->getEndPoint() == lp)
				shouldCurveReverse[i] = 1;
			else
				throw std::exception("create_face: given curves not connected!");
		} // end for i

		// make the face
		BGFace *f = bm_face_create_internal();
		BGLoop *l, *startl, *lastl;
		auto v0 = (BGVert*)(shouldCurveReverse[0] ? curves[0]->getEndPoint() : curves[0]->getStartPoint());
		startl = lastl = bm_face_boundary_add(f, v0, curves[0]);
		startl->v = v0;
		startl->e = curves[0];

		for (int i = 1; i < curves.size(); i++)
		{
			auto v = (BGVert*)(shouldCurveReverse[i] ? curves[i]->getEndPoint() : curves[i]->getStartPoint());
			l = bm_loop_create(v, curves[i], f);
			l->f = f;
			bmesh_radial_append(curves[i], l);

			l->prev = lastl;
			lastl->next = l;
			lastl = l;
		}

		startl->prev = lastl;
		lastl->next = startl;
		f->len = curves.size();
		return f;
	}

	BGFace* BGraph::create_face(BGVert **verts, BGEdge **edges, const int len)
	{
		if (len == 0)
			return 0;

		BGFace *f = bm_face_create_internal();
		BGLoop *l, *startl, *lastl;
		int i;

		startl = lastl = bm_face_boundary_add(f, verts[0], edges[0]);
		startl->v = verts[0];
		startl->e = edges[0];
		for (i = 1; i < len; i++)
		{
			l = bm_loop_create(verts[i], edges[i], f);
			l->f = f;
			bmesh_radial_append(edges[i], l);

			l->prev = lastl;
			lastl->next = l;
			lastl = l;
		}

		startl->prev = lastl;
		lastl->next = startl;
		f->len = len;
		return f;
	}

	BGFace* BGraph::bm_face_create_internal()
	{
		std::shared_ptr<BGFace> f(new BGFace());
		fpool.insert(std::make_pair(f->getId(), f));
		return f.get();
	}

	BGEdge* BGraph::create_edge(const std::vector<std::shared_ptr<BGVert>>& vs)
	{
		std::vector<BGVert*> vptrs;
		for (auto v : vs)
			vptrs.push_back(create_vert(v));
		return create_edge(vptrs);
	}

	BGEdge* BGraph::create_edge(const std::vector<BGVert*>& vs)
	{
		if (vs.size() < 2)
			return 0;
		std::shared_ptr<BGEdge> e(BGEdge::create(vs));
		return create_edge(e);
	}

	BGEdge*	BGraph::create_edge(const std::shared_ptr<BGEdge>& e)
	{
		auto olde = eofv_2((BGVert*)e->getStartPoint(), (BGVert*)e->getEndPoint());
		if (olde)
			return olde;
		epool.insert(std::make_pair(e->getId(), e));
		bmesh_disk_edge_append(e.get(), (BGVert*)e->getStartPoint());
		bmesh_disk_edge_append(e.get(), (BGVert*)e->getEndPoint());
		for (int i = 1; i + 1 < e->numKeyPoints(); i++)
			e->keyPoint(i)->m_edge = e.get();
		return e.get();
	}

	BGLoop* BGraph::bm_loop_create(BGVert *v, BGEdge *e, BGFace *f)
	{
		std::shared_ptr<BGLoop> l(new BGLoop);
		l->next = l->prev = NULL;
		l->v = v;
		l->e = e;
		l->f = f;
		l->radial_next = l->radial_prev = NULL;
		lpool.insert(std::make_pair(l->getId(), l));
		return l.get();
	}

	BGLoop* BGraph::bm_face_boundary_add(BGFace*f, BGVert* startv, BGEdge* starte)
	{
		BGLoop *l = bm_loop_create(startv, starte, f);
		bmesh_radial_append(starte, l);
		f->l_first = l;
		l->f = f;
		return l;
	}

	/**
	 * \brief Split Edge Make Vert (SEMV)
	 *
	 * Takes \a e edge and splits it into two, creating a new vert.
	 * \a tv should be one end of \a e : the newly created edge
	 * will be attached to that end and is returned in \a r_e.
	 *
	 * \par Examples:
	 *
	 *                     E
	 *     Before: OV-------------TV
	 *
	 *                 E       RE
	 *     After:  OV------NV-----TV
	 *
	 * \return The newly created BGVert pointer.
	 */
	BGVert *BGraph::bmesh_semv(BGVert *tv, BGEdge *e, BGEdge **r_e)
	{
		BGLoop *nextl;
		BGEdge *ne;
		BGVert *nv, *ov;
		int i, edok, valence1 = 0, valence2 = 0;

		assert(bmesh_vert_in_edge(e, tv) != FALSE);

		ov = bmesh_edge_other_vert_get(e, tv);

		valence1 = bmesh_disk_count(ov);

		valence2 = bmesh_disk_count(tv);

		nv = create_vert(tv->position());
		std::vector<GraphPoint*> vptrs;
		vptrs.push_back(nv);
		vptrs.push_back(tv);
		ne = create_edge(vptrs);

		bmesh_disk_edge_remove(ne, tv);
		bmesh_disk_edge_remove(ne, nv);

		/* remove e from tv's disk cycle */
		bmesh_disk_edge_remove(e, tv);

		/* swap out tv for nv in e */
		bmesh_edge_swapverts(e, tv, nv);

		/* add e to nv's disk cycle */
		bmesh_disk_edge_append(e, nv);

		/* add ne to nv's disk cycle */
		bmesh_disk_edge_append(ne, nv);

		/* add ne to tv's disk cycle */
		bmesh_disk_edge_append(ne, tv);

		/* verify disk cycle */
		edok = bmesh_disk_validate(valence1, ov->m_edge, ov);
		BGESH_ASSERT(edok != FALSE);
		edok = bmesh_disk_validate(valence2, tv->m_edge, tv);
		BGESH_ASSERT(edok != FALSE);
		edok = bmesh_disk_validate(2, nv->m_edge, nv);
		BGESH_ASSERT(edok != FALSE);

		/* Split the radial cycle if present */
		nextl = e->l;
		e->l = NULL;
		if (nextl)
		{
			BGLoop *nl, *l;
			int radlen = bmesh_radial_length(nextl);
			int first1 = 0, first2 = 0;

			/* Take the next loop. Remove it from radial. Split it. Append to appropriate radials */
			while (nextl)
			{
				l = nextl;
				l->f->len++;
				nextl = nextl != nextl->radial_next ? nextl->radial_next : NULL;
				bmesh_radial_loop_remove(l, NULL);

				nl = bm_loop_create(NULL, NULL, l->f);
				nl->prev = l;
				nl->next = (l->next);
				nl->prev->next = nl;
				nl->next->prev = nl;
				nl->v = nv;

				/* assign the correct edge to the correct loop */
				if (bmesh_verts_in_edge(nl->v, nl->next->v, e))
				{
					nl->e = e;
					l->e = ne;

					/* append l into ne's rad cycle */
					if (!first1)
					{
						first1 = 1;
						l->radial_next = l->radial_prev = NULL;
					}

					if (!first2)
					{
						first2 = 1;
						l->radial_next = l->radial_prev = NULL;
					}

					bmesh_radial_append(nl->e, nl);
					bmesh_radial_append(l->e, l);
				}
				else if (bmesh_verts_in_edge(nl->v, nl->next->v, ne))
				{
					nl->e = ne;
					l->e = e;

					/* append l into ne's rad cycle */
					if (!first1)
					{
						first1 = 1;
						l->radial_next = l->radial_prev = NULL;
					}

					if (!first2)
					{
						first2 = 1;
						l->radial_next = l->radial_prev = NULL;
					}

					bmesh_radial_append(nl->e, nl);
					bmesh_radial_append(l->e, l);
				}

			}

			/* verify length of radial cycle */
			edok = bmesh_radial_validate(radlen, e->l);
			BGESH_ASSERT(edok != FALSE);
			edok = bmesh_radial_validate(radlen, ne->l);
			BGESH_ASSERT(edok != FALSE);

			/* verify loop->v and loop->next->v pointers for e */
			for (i = 0, l = e->l; i < radlen; i++, l = l->radial_next)
			{
				BGESH_ASSERT(l->e == e);
				//BGESH_ASSERT(l->radial_next == l);
				BGESH_ASSERT(!(l->prev->e != ne && l->next->e != ne));

				edok = bmesh_verts_in_edge(l->v, l->next->v, e);
				BGESH_ASSERT(edok != FALSE);
				BGESH_ASSERT(l->v != l->next->v);
				BGESH_ASSERT(l->e != l->next->e);
			}
			/* verify loop->v and loop->next->v pointers for ne */
			for (i = 0, l = ne->l; i < radlen; i++, l = l->radial_next)
			{
				BGESH_ASSERT(l->e == ne);
				// BGESH_ASSERT(l->radial_next == l);
				BGESH_ASSERT(!(l->prev->e != e && l->next->e != e));
				edok = bmesh_verts_in_edge(l->v, l->next->v, ne);
				BGESH_ASSERT(edok != FALSE);
				BGESH_ASSERT(l->v != l->next->v);
				BGESH_ASSERT(l->e != l->next->e);
			}
		}
		if (r_e) *r_e = ne;
		return nv;
	}

	/**
	 * \brief Join Edge Kill Vert (JEKV)
	 *
	 * Takes an edge \a ke and pointer to one of its vertices \a kv
	 * and collapses the edge on that vertex.
	 *
	 * \par Examples:
	 *
	 *     Before:         OE      KE
	 *                   ------- -------
	 *                   |     ||      |
	 *                  OV     KV      TV
	 *
	 *
	 *     After:              OE
	 *                   ---------------
	 *                   |             |
	 *                  OV             TV
	 *
	 * \par Restrictions:
	 * KV is a vertex that must have a valance of exactly two. Furthermore
	 * both edges in KV's disk cycle (OE and KE) must be unique (no double edges).
	 *
	 * \return The resulting edge, NULL for failure.
	 *
	 * \note This euler has the possibility of creating
	 * faces with just 2 edges. It is up to the caller to decide what to do with
	 * these faces.
	 */
	BGEdge *BGraph::bmesh_jekv(BGEdge *ke, BGVert *kv, const short check_edge_double)
	{
		BGEdge *oe;
		BGVert *ov, *tv;
		BGLoop *killoop, *l;
		int len, radlen = 0, halt = 0, i, valence1, valence2, edok;

		if (bmesh_vert_in_edge(ke, kv) == 0)
		{
			return NULL;
		}

		len = bmesh_disk_count(kv);

		if (len == 2)
		{
			oe = bmesh_disk_edge_next(ke, kv);
			tv = bmesh_edge_other_vert_get(ke, kv);
			ov = bmesh_edge_other_vert_get(oe, kv);
			halt = bmesh_verts_in_edge(kv, tv, oe); /* check for double edge */

			if (halt)
			{
				return NULL;
			}
			else
			{
				BGEdge *e_splice;

				/* For verification later, count valence of ov and t */
				valence1 = bmesh_disk_count(ov);
				valence2 = bmesh_disk_count(tv);

				if (check_edge_double)
				{
					e_splice = BG_edge_exists(tv, ov);
				}

				/* remove oe from kv's disk cycle */
				bmesh_disk_edge_remove(oe, kv);
				/* relink oe->kv to be oe->tv */
				bmesh_edge_swapverts(oe, kv, tv);
				/* append oe to tv's disk cycle */
				bmesh_disk_edge_append(oe, tv);
				/* remove ke from tv's disk cycle */
				bmesh_disk_edge_remove(ke, tv);

				/* deal with radial cycle of ke */
				radlen = bmesh_radial_length(ke->l);
				if (ke->l)
				{
					/* first step, fix the neighboring loops of all loops in ke's radial cycle */
					for (i = 0, killoop = ke->l; i < radlen; i++, killoop = killoop->radial_next)
					{
						/* relink loops and fix vertex pointer */
						if (killoop->next->v == kv)
						{
							killoop->next->v = tv;
						}

						killoop->next->prev = killoop->prev;
						killoop->prev->next = killoop->next;
						if (BG_FACE_FIRST_LOOP(killoop->f) == killoop)
						{
							BG_FACE_FIRST_LOOP(killoop->f) = killoop->next;
						}
						killoop->next = NULL;
						killoop->prev = NULL;

						/* fix len attribute of face */
						killoop->f->len--;
					}
					/* second step, remove all the hanging loops attached to ke */
					radlen = bmesh_radial_length(ke->l);

					if (LIKELY(radlen))
					{
						BGLoop **loops = new BGLoop*[radlen];

						killoop = ke->l;

						/* this should be wrapped into a bme_free_radial function to be used by bmesh_KF as well... */
						for (i = 0; i < radlen; i++)
						{
							loops[i] = killoop;
							killoop = killoop->radial_next;
						}
						for (i = 0; i < radlen; i++)
						{
							lpool.erase(loops[i]->getId());
						}
						delete loops;
					}

					/* Validate radial cycle of oe */
					edok = bmesh_radial_validate(radlen, oe->l);
					BGESH_ASSERT(edok != FALSE);
				}

				/* deallocate edg */
				bm_kill_only_edge(this, ke);

				/* deallocate verte */
				bm_kill_only_vert(this, kv);

				/* Validate disk cycle lengths of ov, tv are unchanged */
				edok = bmesh_disk_validate(valence1, ov->m_edge, ov);
				BGESH_ASSERT(edok != FALSE);
				edok = bmesh_disk_validate(valence2, tv->m_edge, tv);
				BGESH_ASSERT(edok != FALSE);

				/* Validate loop cycle of all faces attached to oe */
				for (i = 0, l = oe->l; i < radlen; i++, l = l->radial_next)
				{
					BGESH_ASSERT(l->e == oe);
					edok = bmesh_verts_in_edge(l->v, l->next->v, oe);
					BGESH_ASSERT(edok != FALSE);
					edok = bmesh_loop_validate(l->f);
					BGESH_ASSERT(edok != FALSE);
				}

				if (check_edge_double)
				{
					if (e_splice)
					{
						/* removes e_splice */
						BG_edge_splice(this, e_splice, oe);
					}
				}
				return oe;
			}
		}
		return NULL;
	}

	/**
	 * \brief Split Face Make Edge (SFME)
	 *
	 * Takes as input two vertices in a single face. An edge is created which divides the original face
	 * into two distinct regions. One of the regions is assigned to the original face and it is closed off.
	 * The second region has a new face assigned to it.
	 *
	 * \par Examples:
	 *
	 *     Before:               After:
	 *      +--------+           +--------+
	 *      |        |           |        |
	 *      |        |           |   f1   |
	 *     v1   f1   v2          v1======v2
	 *      |        |           |   f2   |
	 *      |        |           |        |
	 *      +--------+           +--------+
	 *
	 * \note the input vertices can be part of the same edge. This will
	 * result in a two edged face. This is desirable for advanced construction
	 * tools and particularly essential for edge bevel. Because of this it is
	 * up to the caller to decide what to do with the extra edge.
	 *
	 * \note If \a holes is NULL, then both faces will lose
	 * all holes from the original face.  Also, you cannot split between
	 * a hole vert and a boundary vert; that case is handled by higher-
	 * level wrapping functions (when holes are fully implemented, anyway).
	 *
	 * \note that holes represents which holes goes to the new face, and of
	 * course this requires removing them from the existing face first, since
	 * you cannot have linked list links inside multiple lists.
	 *
	 * \return A BGFace pointer
	 */
	BGFace *BGraph::bmesh_sfme(BGFace *f, BGVert *v1, BGVert *v2, BGLoop **r_l)
	{
		BGFace *f2;
		BGLoop *l_iter, *l_first;
		BGLoop *v1loop = NULL, *v2loop = NULL, *f1loop = NULL, *f2loop = NULL;
		BGEdge *e;
		int i, len, f1len, f2len, first_loop_f1;

		/* verify that v1 and v2 are in face */
		len = f->len;
		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f); i < len; i++, l_iter = l_iter->next)
		{
			if (l_iter->v == v1) v1loop = l_iter;
			else if (l_iter->v == v2) v2loop = l_iter;
		}

		if (!v1loop || !v2loop)
		{
			return NULL;
		}

		/* allocate new edge between v1 and v2 */
		std::vector<GraphPoint*> vptrs;
		vptrs.push_back(v1);
		vptrs.push_back(v2);
		e = create_edge(vptrs);

		f2 = bm_face_create_internal();
		f1loop = bm_loop_create(v2, e, f);
		f2loop = bm_loop_create(v1, e, f2);

		f1loop->prev = v2loop->prev;
		f2loop->prev = v1loop->prev;
		v2loop->prev->next = f1loop;
		v1loop->prev->next = f2loop;

		f1loop->next = v1loop;
		f2loop->next = v2loop;
		v1loop->prev = f1loop;
		v2loop->prev = f2loop;

		/* find which of the faces the original first loop is in */
		l_iter = l_first = f1loop;
		first_loop_f1 = 0;
		do
		{
			if (l_iter == f->l_first)
				first_loop_f1 = 1;
		} while ((l_iter = l_iter->next) != l_first);

		if (first_loop_f1)
		{
			/* original first loop was in f1, find a suitable first loop for f2
			 * which is as similar as possible to f1. the order matters for tools
			 * such as duplifaces. */
			if (f->l_first->prev == f1loop)
				f2->l_first = f2loop->prev;
			else if (f->l_first->next == f1loop)
				f2->l_first = f2loop->next;
			else
				f2->l_first = f2loop;
		}
		else
		{
			/* original first loop was in f2, further do same as above */
			f2->l_first = f->l_first;

			if (f->l_first->prev == f2loop)
				f->l_first = f1loop->prev;
			else if (f->l_first->next == f2loop)
				f->l_first = f1loop->next;
			else
				f->l_first = f1loop;
		}

		/* validate both loop */
		/* I don't know how many loops are supposed to be in each face at this point! FIXME */

		/* go through all of f2's loops and make sure they point to it properly */
		l_iter = l_first = BG_FACE_FIRST_LOOP(f2);
		f2len = 0;
		do
		{
			l_iter->f = f2;
			f2len++;
		} while ((l_iter = l_iter->next) != l_first);

		/* link up the new loops into the new edges radial */
		bmesh_radial_append(e, f1loop);
		bmesh_radial_append(e, f2loop);

		f2->len = f2len;

		f1len = 0;
		l_iter = l_first = BG_FACE_FIRST_LOOP(f);
		do
		{
			f1len++;
		} while ((l_iter = l_iter->next) != l_first);

		f->len = f1len;

		if (r_l) *r_l = f2loop;

		return f2;
	}

	/**
	 * \brief Face Split
	 *
	 * Split a face along two vertices. returns the newly made face, and sets
	 * the \a r_l member to a loop in the newly created edge.
	 *
	 * \param bm The bmesh
	 * \param f the original face
	 * \param v1, v2 vertices which define the split edge, must be different
	 * \param r_l pointer which will receive the BGLoop for the split edge in the new face
	 * \param example Edge used for attributes of splitting edge, if non-NULL
	 * \param nodouble Use an existing edge if found
	 *
	 * \return Pointer to the newly created face representing one side of the split
	 * if the split is successful (and the original original face will be the
	 * other side). NULL if the split fails.
	 */
	BGFace *BGraph::BG_face_split(BGFace *f, BGVert *v1, BGVert *v2, BGLoop **r_l)
	{
		BGFace *nf;

		assert(v1 != v2);

		nf = bmesh_sfme(f, v1, v2, r_l);

		return nf;
	}

	/**
	 * \brief Join Face Kill Edge (JFKE)
	 *
	 * Takes two faces joined by a single 2-manifold edge and fuses them together.
	 * The edge shared by the faces must not be connected to any other edges which have
	 * Both faces in its radial cycle
	 *
	 * \par Examples:
	 *
	 *           A                   B
	 *      +--------+           +--------+
	 *      |        |           |        |
	 *      |   f1   |           |   f1   |
	 *     v1========v2 = Ok!    v1==V2==v3 == Wrong!
	 *      |   f2   |           |   f2   |
	 *      |        |           |        |
	 *      +--------+           +--------+
	 *
	 * In the example A, faces \a f1 and \a f2 are joined by a single edge,
	 * and the euler can safely be used.
	 * In example B however, \a f1 and \a f2 are joined by multiple edges and will produce an error.
	 * The caller in this case should call #bmesh_jekv on the extra edges
	 * before attempting to fuse \a f1 and \a f2.
	 *
	 * \note The order of arguments decides whether or not certain per-face attributes are present
	 * in the resultant face. For instance vertex winding, material index, smooth flags, etc are inherited
	 * from \a f1, not \a f2.
	 *
	 * \return A BGFace pointer
	 */
	BGFace *BGraph::bmesh_jfke(BGFace *f1, BGFace *f2, BGEdge *e)
	{
		BGLoop *l_iter, *f1loop = NULL, *f2loop = NULL;
		int newlen = 0, i, f1len = 0, f2len = 0, edok;

		/* can't join a face to itself */
		if (f1 == f2)
		{
			return NULL;
		}

		/* validate that edge is 2-manifold edge */
		if (!BG_edge_is_manifold(e))
		{
			return NULL;
		}

		/* verify that e is in both f1 and f2 */
		f1len = f1->len;
		f2len = f2->len;

		if (!((f1loop = BG_face_edge_share_loop(f1, e)) &&
			(f2loop = BG_face_edge_share_loop(f2, e))))
		{
			return NULL;
		}

		/* validate direction of f2's loop cycle is compatible */
		if (f1loop->v == f2loop->v)
		{
			return NULL;
		}

		/* validate that for each face, each vertex has another edge in its disk cycle that is
		 * not e, and not shared. */
		if (bmesh_radial_face_find(f1loop->next->e, f2) ||
			bmesh_radial_face_find(f1loop->prev->e, f2) ||
			bmesh_radial_face_find(f2loop->next->e, f1) ||
			bmesh_radial_face_find(f2loop->prev->e, f1))
		{
			return NULL;
		}

		/* validate only one shared edge */
		if (BG_face_share_edge_count(f1, f2) > 1)
		{
			return NULL;
		}

		/* validate no internal join */
		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f1); i < f1len; i++, l_iter = l_iter->next)
		{
			l_iter->v->m_flag = 0;
		}
		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f2); i < f2len; i++, l_iter = l_iter->next)
		{
			l_iter->v->m_flag = 0;
		}

		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f1); i < f1len; i++, l_iter = l_iter->next)
		{
			if (l_iter != f1loop)
			{
				l_iter->v->m_flag = 1;
			}
		}
		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f2); i < f2len; i++, l_iter = l_iter->next)
		{
			if (l_iter != f2loop)
			{
				/* as soon as a duplicate is found, bail out */
				if (l_iter->v->m_flag == 1)
				{
					return NULL;
				}
			}
		}

		/* join the two loop */
		f1loop->prev->next = f2loop->next;
		f2loop->next->prev = f1loop->prev;

		f1loop->next->prev = f2loop->prev;
		f2loop->prev->next = f1loop->next;

		/* if f1loop was baseloop, make f1loop->next the base. */
		if (BG_FACE_FIRST_LOOP(f1) == f1loop)
			BG_FACE_FIRST_LOOP(f1) = f1loop->next;

		/* increase length of f1 */
		f1->len += (f2->len - 2);

		/* make sure each loop points to the proper face */
		newlen = f1->len;
		for (i = 0, l_iter = BG_FACE_FIRST_LOOP(f1); i < newlen; i++, l_iter = l_iter->next)
			l_iter->f = f1;

		/* remove edge from the disk cycle of its two vertices */
		bmesh_disk_edge_remove(f1loop->e, (BGVert*)f1loop->e->getStartPoint());
		bmesh_disk_edge_remove(f1loop->e, (BGVert*)f1loop->e->getEndPoint());

		/* deallocate edge and its two loops as well as f2 */
		epool.erase(f1loop->e->getId());
		lpool.erase(f1loop->getId());
		lpool.erase(f2loop->getId());
		fpool.erase(f2->getId());

		/* validate the new loop cycle */
		edok = bmesh_loop_validate(f1);
		BGESH_ASSERT(edok != FALSE);

		return f1;
	}

	/************************************************************************/
	/* static functions
	/************************************************************************/
	bool BGraph::bmesh_disk_edge_append(BGEdge* e, BGVert* v)
	{
		if (!v->m_edge)
		{
			BGDiskLink *dl1 = BG_DISK_EDGE_LINK_GET(e, v);

			v->m_edge = e;
			dl1->next = dl1->prev = e;
		}
		else
		{
			BGDiskLink *dl1, *dl2, *dl3;

			dl1 = BG_DISK_EDGE_LINK_GET(e, v);
			dl2 = BG_DISK_EDGE_LINK_GET(v->m_edge, v);
			dl3 = dl2->prev ? BG_DISK_EDGE_LINK_GET(dl2->prev, v) : NULL;

			dl1->next = v->m_edge;
			dl1->prev = dl2->prev;

			dl2->prev = e;
			if (dl3)
				dl3->next = e;
		}

		return true;
	}

	void BGraph::bmesh_radial_append(BGEdge* e, BGLoop* l)
	{
		if (e->l == 0)
		{
			e->l = l;
			l->radial_next = l->radial_prev = l;
		}
		else
		{
			l->radial_prev = e->l;
			l->radial_next = e->l->radial_next;

			e->l->radial_next->radial_prev = l;
			e->l->radial_next = l;

			e->l = l;
		}

		if (l->e && l->e != e)
		{
			assert(0);
		}

		l->e = e;
	}

	int BGraph::bmesh_radial_length(const BGLoop *l)
	{
		const BGLoop *l_iter = l;
		int i = 0;

		if (!l)
			return 0;

		do
		{
			if (!l_iter)
			{
				/* radial cycle is broken (not a circulat loop) */
				assert(0);
				return 0;
			}

			i++;
			if (i >= BG_LOOP_RADIAL_MAX)
			{
				assert(0);
				return -1;
			}
		} while ((l_iter = l_iter->radial_next) != l);

		return i;
	}

	/**
	 * \brief Next Disk Edge
	 *
	 *	Find the next edge in a disk cycle
	 *
	 *	\return Pointer to the next edge in the disk cycle for the vertex v.
	 */
	BGEdge *BGraph::bmesh_disk_edge_next(const BGEdge *e, const BGVert *v)
	{
		if (v == e->getStartPoint())
			return e->v1_disk_link.next;
		if (v == e->getEndPoint())
			return e->v2_disk_link.next;
		return NULL;
	}

	BGEdge *BGraph::bmesh_disk_edge_prev(const BGEdge *e, const BGVert *v)
	{
		if (v == e->getStartPoint())
			return e->v1_disk_link.prev;
		if (v == e->getEndPoint())
			return e->v2_disk_link.prev;
		return NULL;
	}

	BGEdge *BGraph::bmesh_disk_edge_exists(const BGVert *v1, const BGVert *v2)
	{
		BGEdge *e_iter, *e_first;

		if (v1->m_edge)
		{
			e_first = e_iter = v1->m_edge;

			do
			{
				if (bmesh_verts_in_edge(v1, v2, e_iter))
				{
					return e_iter;
				}
			} while ((e_iter = bmesh_disk_edge_next(e_iter, v1)) != e_first);
		}

		return NULL;
	}

	int BGraph::bmesh_disk_count(const BGVert *v)
	{
		if (v->m_edge)
		{
			BGEdge *e_first, *e_iter;
			int count = 0;

			e_iter = e_first = v->m_edge;

			do
			{
				if (!e_iter)
				{
					return 0;
				}

				if (count >= (1 << 20))
				{
					printf("bmesh error: infinite loop in disk cycle!\n");
					return 0;
				}
				count++;
			} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e_first);
			return count;
		}
		else
		{
			return 0;
		}
	}

	int BGraph::bmesh_disk_validate(int len, const BGEdge *e, const BGVert *v)
	{
		const BGEdge *e_iter;

		if (!BG_vert_in_edge(e, v))
			return 0;
		if (bmesh_disk_count(v) != len || len == 0)
			return 0;

		e_iter = e;
		do
		{
			if (len != 1 && bmesh_disk_edge_prev(e_iter, v) == e_iter)
			{
				return 0;
			}
		} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e);

		return 0;
	}

	/**
	 *	MISC utility functions.
	 */

	int BGraph::bmesh_vert_in_edge(const BGEdge *e, const BGVert *v)
	{
		if (e->getStartPoint() == v || e->getEndPoint() == v) return 1;
		return 0;
	}
	int BGraph::bmesh_verts_in_edge(const BGVert *v1, const BGVert *v2, const BGEdge *e)
	{
		if (e->getStartPoint() == v1 && e->getEndPoint() == v2) return 1;
		else if (e->getStartPoint() == v2 && e->getEndPoint() == v1) return 1;
		return 0;
	}

	BGVert *BGraph::bmesh_edge_other_vert_get(BGEdge *e, const BGVert *v)
	{
		if (e->getStartPoint() == v)
		{
			return (BGVert*)e->getEndPoint();
		}
		else if (e->getEndPoint() == v)
		{
			return (BGVert*)e->getStartPoint();
		}
		return 0;
	}

	int BGraph::bmesh_edge_swapverts(BGEdge *e, BGVert *orig, BGVert *newv)
	{
		if (e->getStartPoint() == orig)
		{
			e->keyPoint(0) = newv;
			e->v1_disk_link.next = e->v1_disk_link.prev = 0;
			return 1;
		}
		else if (e->getEndPoint() == orig)
		{
			e->keyPoint(e->numKeyPoints() - 1) = newv;
			e->v2_disk_link.next = e->v2_disk_link.prev = 0;
			return 1;
		}
		return 0;
	}

	void BGraph::bmesh_disk_edge_remove(BGEdge *e, BGVert *v)
	{
		BGDiskLink *dl1, *dl2;

		dl1 = BG_DISK_EDGE_LINK_GET(e, v);
		if (dl1->prev)
		{
			dl2 = BG_DISK_EDGE_LINK_GET(dl1->prev, v);
			dl2->next = dl1->next;
		}

		if (dl1->next)
		{
			dl2 = BG_DISK_EDGE_LINK_GET(dl1->next, v);
			dl2->prev = dl1->prev;
		}

		if (v->m_edge == e)
			v->m_edge = (e != dl1->next) ? dl1->next : NULL;

		dl1->next = dl1->prev = NULL;
	}

	/**
	 * \brief DISK COUNT FACE VERT
	 *
	 * Counts the number of loop users
	 * for this vertex. Note that this is
	 * equivalent to counting the number of
	 * faces incident upon this vertex
	 */
	int BGraph::bmesh_disk_facevert_count(const BGVert *v)
	{
		/* is there an edge on this vert at all */
		if (v->m_edge)
		{
			BGEdge *e_first, *e_iter;
			int count = 0;

			/* first, loop around edge */
			e_first = e_iter = v->m_edge;
			do
			{
				if (e_iter->l)
				{
					count += bmesh_radial_facevert_count(e_iter->l, v);
				}
			} while ((e_iter = bmesh_disk_edge_next(e_iter, v)) != e_first);
			return count;
		}
		else
		{
			return 0;
		}
	}

	/**
	 * \brief FIND FIRST FACE EDGE
	 *
	 * Finds the first edge in a vertices
	 * Disk cycle that has one of this
	 * vert's loops attached
	 * to it.
	 */
	BGEdge *BGraph::bmesh_disk_faceedge_find_first(BGEdge *e, const BGVert *v)
	{
		BGEdge *searchedge = NULL;
		searchedge = e;
		do
		{
			if (searchedge->l && bmesh_radial_facevert_count(searchedge->l, v))
			{
				return searchedge;
			}
		} while ((searchedge = bmesh_disk_edge_next(searchedge, v)) != e);

		return NULL;
	}

	BGEdge *BGraph::bmesh_disk_faceedge_find_next(BGEdge *e, const BGVert *v)
	{
		BGEdge *searchedge = NULL;
		searchedge = bmesh_disk_edge_next(e, v);
		do
		{
			if (searchedge->l && bmesh_radial_facevert_count(searchedge->l, v))
			{
				return searchedge;
			}
		} while ((searchedge = bmesh_disk_edge_next(searchedge, v)) != e);
		return e;
	}

	/*****radial cycle functions, e.g. loops surrounding edges**** */
	int BGraph::bmesh_radial_validate(int radlen, const BGLoop *l)
	{
		const BGLoop *l_iter = l;
		int i = 0;

		if (bmesh_radial_length(l) != radlen)
			return FALSE;

		do
		{
			if (UNLIKELY(!l_iter))
			{
				BGESH_ASSERT(0);
				return FALSE;
			}

			if (l_iter->e != l->e)
				return FALSE;
			if (l_iter->v != l->e->getStartPoint() && l_iter->v != l->e->getEndPoint())
				return FALSE;

			if (UNLIKELY(i > BG_LOOP_RADIAL_MAX))
			{
				BGESH_ASSERT(0);
				return FALSE;
			}

			i++;
		} while ((l_iter = l_iter->radial_next) != l);

		return TRUE;
	}

	/**
	 * \brief BGESH RADIAL REMOVE LOOP
	 *
	 * Removes a loop from an radial cycle. If edge e is non-NULL
	 * it should contain the radial cycle, and it will also get
	 * updated (in the case that the edge's link into the radial
	 * cycle was the loop which is being removed from the cycle).
	 */
	void BGraph::bmesh_radial_loop_remove(BGLoop *l, BGEdge *e)
	{
		/* if e is non-NULL, l must be in the radial cycle of e */
		if (UNLIKELY(e && e != l->e))
		{
			assert(0);
		}

		if (l->radial_next != l)
		{
			if (e && l == e->l)
				e->l = l->radial_next;

			l->radial_next->radial_prev = l->radial_prev;
			l->radial_prev->radial_next = l->radial_next;
		}
		else
		{
			if (e)
			{
				if (l == e->l)
				{
					e->l = NULL;
				}
				else
				{
					BGESH_ASSERT(0);
				}
			}
		}

		/* l is no longer in a radial cycle; empty the links
		 * to the cycle and the link back to an edge */
		l->radial_next = l->radial_prev = NULL;
		l->e = NULL;
	}


	/**
	 * \brief BGE RADIAL FIND FIRST FACE VERT
	 *
	 * Finds the first loop of v around radial
	 * cycle
	 */
	BGLoop *BGraph::bmesh_radial_faceloop_find_first(BGLoop *l, const BGVert *v)
	{
		BGLoop *l_iter;
		l_iter = l;
		do
		{
			if (l_iter->v == v)
			{
				return l_iter;
			}
		} while ((l_iter = l_iter->radial_next) != l);
		return NULL;
	}

	BGLoop *BGraph::bmesh_radial_faceloop_find_next(BGLoop *l, const BGVert *v)
	{
		BGLoop *l_iter;
		l_iter = l->radial_next;
		do
		{
			if (l_iter->v == v)
			{
				return l_iter;
			}
		} while ((l_iter = l_iter->radial_next) != l);
		return l;
	}

	int BGraph::bmesh_radial_face_find(const BGEdge *e, const BGFace *f)
	{
		const BGLoop *l_iter;
		int i, len;

		len = bmesh_radial_length(e->l);
		for (i = 0, l_iter = e->l; i < len; i++, l_iter = l_iter->radial_next)
		{
			if (l_iter->f == f)
				return TRUE;
		}
		return FALSE;
	}

	/**
	 * \brief RADIAL COUNT FACE VERT
	 *
	 * Returns the number of times a vertex appears
	 * in a radial cycle
	 */
	int BGraph::bmesh_radial_facevert_count(const BGLoop *l, const BGVert *v)
	{
		const BGLoop *l_iter = l;
		int count = 0;
		do
		{
			if (l_iter->v == v)
			{
				count++;
			}
		} while ((l_iter = l_iter->radial_next) != l);

		return count;
	}

	/*****loop cycle functions, e.g. loops surrounding a face**** */
	int BGraph::bmesh_loop_validate(const BGFace *f)
	{
		int i;
		int len = f->len;
		BGLoop *l_iter, *l_first;

		l_first = BG_FACE_FIRST_LOOP(f);

		if (l_first == NULL)
		{
			return FALSE;
		}

		/* Validate that the face loop cycle is the length specified by f->len */
		for (i = 1, l_iter = l_first->next; i < len; i++, l_iter = l_iter->next)
		{
			if ((l_iter->f != f) ||
				(l_iter == l_first))
			{
				return FALSE;
			}
		}
		if (l_iter != l_first)
		{
			return FALSE;
		}

		/* Validate the loop->prev links also form a cycle of length f->len */
		for (i = 1, l_iter = l_first->prev; i < len; i++, l_iter = l_iter->prev)
		{
			if (l_iter == l_first)
			{
				return FALSE;
			}
		}
		if (l_iter != l_first)
		{
			return FALSE;
		}

		return TRUE;
	}

	int BGraph::bm_loop_length(BGLoop *l)
	{
		BGLoop *l_first = l;
		int i = 0;

		do
		{
			i++;
		} while ((l = l->next) != l_first);

		return i;
	}


	int	BGraph::bmesh_loop_reverse(BGraph* bm, BGFace* f)
	{
		BGLoop *l_first = f->l_first;

		BGLoop *l_iter, *oldprev, *oldnext;
		vector<BGEdge*> edar;
		edar.reserve(BG_NGON_STACK_SIZE);

		int i, j, edok, len = 0;

		len = bm_loop_length(l_first);

		for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next)
		{
			BGEdge *curedge = l_iter->e;
			bmesh_radial_loop_remove(l_iter, curedge);
			edar.push_back(curedge);
		}

		/* actually reverse the loop */
		for (i = 0, l_iter = l_first; i < len; i++)
		{
			oldnext = l_iter->next;
			oldprev = l_iter->prev;
			l_iter->next = oldprev;
			l_iter->prev = oldnext;
			l_iter = oldnext;
		}

		if (len == 2)
		{ /* two edged face */
			/* do some verification here! */
			l_first->e = edar[1];
			l_first->next->e = edar[0];
		}
		else
		{
			for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next)
			{
				edok = 0;
				for (j = 0; j < len; j++)
				{
					edok = bmesh_verts_in_edge(l_iter->v, l_iter->next->v, edar[j]);
					if (edok)
					{
						l_iter->e = edar[j];
						break;
					}
				}
			}
		}
		/* rebuild radia */
		for (i = 0, l_iter = l_first; i < len; i++, l_iter = l_iter->next)
			bmesh_radial_append(l_iter->e, l_iter);

		return 1;
	}


#define BG_OVERLAP (1 << 13)

	/**
	 * Returns whether or not a given vertex is
	 * is part of a given edge.
	 */
	int BGraph::BG_vert_in_edge(const BGEdge *e, const BGVert *v)
	{
		return bmesh_vert_in_edge(e, v);
	}

	/**
	 * \brief Other Loop in Face Sharing an Edge
	 *
	 * Finds the other loop that shares \a v with \a e loop in \a f.
	 *
	 *     +----------+
	 *     |          |
	 *     |    f     |
	 *     |          |
	 *     +----------+ <-- return the face loop of this vertex.
	 *     v --> e
	 *     ^     ^ <------- These vert args define direction
	 *                      in the face to check.
	 *                      The faces loop direction is ignored.
	 *
	 */
	BGLoop *BGraph::BG_face_other_edge_loop(BGFace *f, BGEdge *e, BGVert *v)
	{
		BGLoop *l_iter;
		BGLoop *l_first;

		/* we could loop around the face too, but turns out this uses a lot
		 * more iterations (approx double with quads, many more with 5+ ngons) */
		l_iter = l_first = e->l;

		do
		{
			if (l_iter->e == e && l_iter->f == f)
			{
				break;
			}
		} while ((l_iter = l_iter->radial_next) != l_first);

		return l_iter->v == v ? l_iter->prev : l_iter->next;
	}

	/**
	 * \brief Other Loop in Face Sharing a Vertex
	 *
	 * Finds the other loop in a face.
	 *
	 * This function returns a loop in \a f that shares an edge with \a v
	 * The direction is defined by \a v_prev, where the return value is
	 * the loop of what would be 'v_next'
	 *
	 *
	 *     +----------+ <-- return the face loop of this vertex.
	 *     |          |
	 *     |    f     |
	 *     |          |
	 *     +----------+
	 *     v_prev --> v
	 *     ^^^^^^     ^ <-- These vert args define direction
	 *                      in the face to check.
	 *                      The faces loop direction is ignored.
	 *
	 * \note \a v_prev and \a v _implicitly_ define an edge.
	 */
	BGLoop *BGraph::BG_face_other_vert_loop(BGFace *f, BGVert *v_prev, BGVert *v)
	{
		BGIter liter;
		BGLoop *l_iter;

		assert(BG_edge_exists(v_prev, v) != NULL);
		liter.init(f);
		for (l_iter = lofv_begin(liter); l_iter != lofv_end(liter); l_iter = lofv_next(liter))
		{
			if (l_iter->f == f) break;
		}

		if (l_iter)
		{
			if (l_iter->prev->v == v_prev)
			{
				return l_iter->next;
			}
			else if (l_iter->next->v == v_prev)
			{
				return l_iter->prev;
			}
			else
			{
				/* invalid args */
				assert(0);
				return NULL;
			}
		}
		else
		{
			/* invalid args */
			assert(0);
			return NULL;
		}
	}

	/**
	 * \brief Other Loop in Face Sharing a Vert
	 *
	 * Finds the other loop that shares \a v with \a e loop in \a f.
	 *
	 *     +----------+ <-- return the face loop of this vertex.
	 *     |          |
	 *     |          |
	 *     |          |
	 *     +----------+ <-- This vertex defines the direction.
	 *           l    v
	 *           ^ <------- This loop defines both the face to search
	 *                      and the edge, in combination with 'v'
	 *                      The faces loop direction is ignored.
	 */

	BGLoop *BGraph::BG_loop_other_vert_loop(BGLoop *l, BGVert *v)
	{
		BGEdge *e = l->e;
		BGVert *v_prev = BG_edge_other_vert(e, v);
		if (l->v == v)
		{
			if (l->prev->v == v_prev)
			{
				return l->next;
			}
			else
			{
				assert(l->next->v == v_prev);

				return l->prev;
			}
		}
		else
		{
			assert(l->v == v_prev);

			if (l->prev->v == v)
			{
				return l->prev->prev;
			}
			else
			{
				assert(l->next->v == v);
				return l->next->next;
			}
		}
	}

	/**
	 * Returns TRUE if the vertex is used in a given face.
	 */

	int BGraph::BG_vert_in_face(BGFace *f, BGVert *v)
	{
		BGLoop *l_iter, *l_first;
		l_iter = l_first = f->l_first;
		do
		{
			if (l_iter->v == v)
			{
				return TRUE;
			}
		} while ((l_iter = l_iter->next) != l_first);

		return FALSE;
	}

	/**
	 * Compares the number of vertices in an array
	 * that appear in a given face
	 */
	int BGraph::BG_verts_in_face(BGFace *f, BGVert **varr, int len)
	{
		BGLoop *l_iter, *l_first;

		int count = 0;

		l_iter = l_first = f->l_first;
		do
		{
			for (int i = 0; i < len; i++)
			{
				if (varr[i] == l_iter->v)
				{
					count++;
				}
			}

		} while ((l_iter = l_iter->next) != l_first);

		return count;
	}

	/**
	 * Returns whether or not a given edge is is part of a given face.
	 */
	int BGraph::BG_edge_in_face(BGFace *f, BGEdge *e)
	{
		BGLoop *l_iter;
		BGLoop *l_first;

		l_iter = l_first = f->l_first;

		do
		{
			if (l_iter->e == e)
			{
				return TRUE;
			}
		} while ((l_iter = l_iter->next) != l_first);

		return FALSE;
	}

	/**
	 * Returns whether or not two vertices are in
	 * a given edge
	 */
	int BGraph::BG_verts_in_edge(BGVert *v1, BGVert *v2, BGEdge *e)
	{
		return bmesh_verts_in_edge(v1, v2, e);
	}

	/**
	 * Given a edge and one of its vertices, returns
	 * the other vertex.
	 */
	BGVert *BGraph::BG_edge_other_vert(BGEdge *e, BGVert *v)
	{
		return bmesh_edge_other_vert_get(e, v);
	}

	/**
	 * The function takes a vertex at the center of a fan and returns the opposite edge in the fan.
	 * All edges in the fan must be manifold, otherwise return NULL.
	 *
	 * \note This could (probably) be done more effieiently.
	 */
	BGEdge *BGraph::BG_vert_other_disk_edge(BGVert *v, BGEdge *e_first)
	{
		BGLoop *l_a;
		int tot = 0;
		int i;

		assert(BG_vert_in_edge(e_first, v));

		l_a = e_first->l;
		do
		{
			l_a = BG_loop_other_vert_loop(l_a, v);
			l_a = BG_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
			if (BG_edge_is_manifold(l_a->e))
			{
				l_a = l_a->radial_next;
			}
			else
			{
				return NULL;
			}

			tot++;
		} while (l_a != e_first->l);

		/* we know the total, now loop half way */
		tot /= 2;
		i = 0;

		l_a = e_first->l;
		do
		{
			if (i == tot)
			{
				l_a = BG_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
				return l_a->e;
			}

			l_a = BG_loop_other_vert_loop(l_a, v);
			l_a = BG_vert_in_edge(l_a->e, v) ? l_a : l_a->prev;
			if (BG_edge_is_manifold(l_a->e))
			{
				l_a = l_a->radial_next;
			}
			/* this wont have changed from the previous loop */


			i++;
		} while (l_a != e_first->l);

		return NULL;
	}

	/**
	 * Returms edge length
	 */
	float BGraph::BG_edge_calc_length(BGEdge *e)
	{
		return e->getLength();
	}

	/**
	 * Utility function, since enough times we have an edge
	 * and want to access 2 connected faces.
	 *
	 * \return TRUE when only 2 faces are found.
	 */
	int BGraph::BG_edge_face_pair(BGEdge *e, BGFace **r_fa, BGFace **r_fb)
	{
		BGLoop *la, *lb;

		if ((la = e->l) &&
			(lb = la->radial_next) &&
			(lb->radial_next == la))
		{
			*r_fa = la->f;
			*r_fb = lb->f;
			return TRUE;
		}
		else
		{
			*r_fa = NULL;
			*r_fb = NULL;
			return FALSE;
		}
	}

	/**
	 * Utility function, since enough times we have an edge
	 * and want to access 2 connected loops.
	 *
	 * \return TRUE when only 2 faces are found.
	 */
	int BGraph::BG_edge_loop_pair(BGEdge *e, BGLoop **r_la, BGLoop **r_lb)
	{
		BGLoop *la, *lb;

		if ((la = e->l) &&
			(lb = la->radial_next) &&
			(lb->radial_next == la))
		{
			*r_la = la;
			*r_lb = lb;
			return TRUE;
		}
		else
		{
			*r_la = NULL;
			*r_lb = NULL;
			return FALSE;
		}
	}

	/**
	 *	Returns the number of edges around this vertex.
	 */
	int BGraph::BG_vert_edge_count(BGVert *v)
	{
		return bmesh_disk_count(v);
	}

	int BGraph::BG_vert_edge_count_nonwire(BGVert *v)
	{
		int count = 0;
		BGIter eiter;
		BGEdge *edge;
		eiter.init(v);
		for (edge = eofv_begin(eiter); edge != eofv_end(eiter); edge = eofv_next(eiter))
		{
			if (edge->l)	count++;
		}
		return count;
	}
	/**
	 *	Returns the number of faces around this edge
	 */
	int BGraph::BG_edge_face_count(BGEdge *e)
	{
		int count = 0;

		if (e->l)
		{
			BGLoop *l_iter;
			BGLoop *l_first;

			l_iter = l_first = e->l;

			do
			{
				count++;
			} while ((l_iter = l_iter->radial_next) != l_first);
		}

		return count;
	}

	/**
	 *	Returns the number of faces around this vert
	 */
	int BGraph::BG_vert_face_count(BGVert *v)
	{
		int count = 0;
		BGLoop *l;
		BGIter iter;
		iter.init(v);
		for (l = lofv_begin(iter); l != lofv_end(iter); l = lofv_next(iter))
			count++;
		return count;
	}

	/**
	 * Tests whether or not the vertex is part of a wire edge.
	 * (ie: has no faces attached to it)
	 */
	int BGraph::BG_vert_is_wire(BGVert *v)
	{
		BGEdge *curedge;

		if (v->m_edge == NULL)
		{
			return FALSE;
		}

		curedge = v->m_edge;
		do
		{
			if (curedge->l)
			{
				return FALSE;
			}

			curedge = bmesh_disk_edge_next(curedge, v);
		} while (curedge != v->m_edge);

		return TRUE;
	}

	/**
	 * Tests whether or not the edge is part of a wire.
	 * (ie: has no faces attached to it)
	 */
	int BGraph::BG_edge_is_wire(BGEdge *e)
	{
		return (e->l) ? FALSE : TRUE;
	}

	/**
	 * A vertex is non-manifold if it meets the following conditions:
	 * 1: Loose - (has no edges/faces incident upon it).
	 * 2: Joins two distinct regions - (two pyramids joined at the tip).
	 * 3: Is part of a an edge with more than 2 faces.
	 * 4: Is part of a wire edge.
	 */
	int BGraph::BG_vert_is_manifold(BGVert *v)
	{
		BGEdge *e, *oe;
		BGLoop *l;
		int len, count, flag;

		if (v->m_edge == NULL)
		{
			/* loose vert */
			return FALSE;
		}

		/* count edges while looking for non-manifold edges */
		len = 0;
		oe = e = v->m_edge;
		do
		{
			/* loose edge or edge shared by more than two faces,
			 * edges with 1 face user are OK, otherwise we could
			 * use BG_edge_is_manifold() here */
			if (e->l == NULL || bmesh_radial_length(e->l) > 2)
			{
				return FALSE;
			}
			len++;
		} while ((e = bmesh_disk_edge_next(e, v)) != oe);

		count = 1;
		flag = 1;
		e = NULL;
		oe = v->m_edge;
		l = oe->l;
		while (e != oe)
		{
			l = (l->v == v) ? l->prev : l->next;
			e = l->e;
			count++; /* count the edges */

			if (flag && l->radial_next == l)
			{
				/* we've hit the edge of an open mesh, reset once */
				flag = 0;
				count = 1;
				oe = e;
				e = NULL;
				l = oe->l;
			}
			else if (l->radial_next == l)
			{
				/* break the loop */
				e = oe;
			}
			else
			{
				l = l->radial_next;
			}
		}

		if (count < len)
		{
			/* vert shared by multiple regions */
			return FALSE;
		}

		return TRUE;
	}

	/**
	 * Tests whether or not this edge is manifold.
	 * A manifold edge has exactly 2 faces attached to it.
	 */

#if 1 /* fast path for checking manifold */
	int BGraph::BG_edge_is_manifold(BGEdge *e)
	{
		const BGLoop *l = e->l;
		return (l && (l->radial_next != l) &&             /* not 0 or 1 face users */
			(l->radial_next->radial_next == l)); /* 2 face users */
	}
#else
	int BG_edge_is_manifold(BGEdge *e)
	{
		int count = BG_edge_face_count(e);
		if (count == 2) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
#endif

	/**
	 * Tests whether or not an edge is on the boundary
	 * of a shell (has one face associated with it)
	 */

#if 1 /* fast path for checking boundary */
	int BGraph::BG_edge_is_boundary(BGEdge *e)
	{
		const BGLoop *l = e->l;
		return (l && (l->radial_next == l));
	}
#else
	int BG_edge_is_boundary(BGEdge *e)
	{
		int count = BG_edge_face_count(e);
		if (count == 1) {
			return TRUE;
		}
		else {
			return FALSE;
		}
	}
#endif

	/**
	 *  Counts the number of edges two faces share (if any)
	 */
	int BGraph::BG_face_share_edge_count(BGFace *f1, BGFace *f2)
	{
		BGLoop *l_iter;
		BGLoop *l_first;
		int count = 0;

		l_iter = l_first = BG_FACE_FIRST_LOOP(f1);
		do
		{
			if (bmesh_radial_face_find(l_iter->e, f2))
			{
				count++;
			}
		} while ((l_iter = l_iter->next) != l_first);

		return count;
	}

	/**
	 *	Test if e1 shares any faces with e2
	 */
	int BGraph::BG_edge_share_face_count(BGEdge *e1, BGEdge *e2)
	{
		BGLoop *l;
		BGFace *f;

		if (e1->l && e2->l)
		{
			l = e1->l;
			do
			{
				f = l->f;
				if (bmesh_radial_face_find(e2, f))
				{
					return TRUE;
				}
				l = l->radial_next;
			} while (l != e1->l);
		}
		return FALSE;
	}

	/**
	 *	Tests to see if e1 shares a vertex with e2
	 */
	int BGraph::BG_edge_share_vert_count(BGEdge *e1, BGEdge *e2)
	{
		return (e1->getStartPoint() == e2->getStartPoint() ||
			e1->getStartPoint() == e2->getEndPoint() ||
			e1->getEndPoint() == e2->getStartPoint() ||
			e1->getEndPoint() == e2->getEndPoint());
	}

	/**
	 *	Return the shared vertex between the two edges or NULL
	 */
	BGVert *BGraph::BG_edge_share_vert(BGEdge *e1, BGEdge *e2)
	{
		if (BG_vert_in_edge(e2, e1->getStartPoint()))
		{
			return (BGVert*)e1->getStartPoint();
		}
		else if (BG_vert_in_edge(e2, e1->getEndPoint()))
		{
			return (BGVert*)e1->getEndPoint();
		}
		else
		{
			return NULL;
		}
	}

	/**
	 * \brief Return the Loop Shared by Face and Vertex
	 *
	 * Finds the loop used which uses \a v in face loop \a l
	 *
	 * \note currenly this just uses simple loop in future may be speeded up
	 * using radial vars
	 */
	BGLoop *BGraph::BG_face_vert_share_loop(BGFace *f, BGVert *v)
	{
		BGLoop *l_first;
		BGLoop *l_iter;

		l_iter = l_first = BG_FACE_FIRST_LOOP(f);
		do
		{
			if (l_iter->v == v)
			{
				return l_iter;
			}
		} while ((l_iter = l_iter->next) != l_first);

		return NULL;
	}

	/**
	 * \brief Return the Loop Shared by Face and Edge
	 *
	 * Finds the loop used which uses \a e in face loop \a l
	 *
	 * \note currenly this just uses simple loop in future may be speeded up
	 * using radial vars
	 */
	BGLoop *BGraph::BG_face_edge_share_loop(BGFace *f, BGEdge *e)
	{
		BGLoop *l_first;
		BGLoop *l_iter;

		l_iter = l_first = e->l;
		do
		{
			if (l_iter->f == f)
			{
				return l_iter;
			}
		} while ((l_iter = l_iter->radial_next) != l_first);

		return NULL;
	}

	/**
	 * Returns the verts of an edge as used in a face
	 * if used in a face at all, otherwise just assign as used in the edge.
	 *
	 * Useful to get a deterministic winding order when calling
	 * BG_face_create_ngon() on an arbitrary array of verts,
	 * though be sure to pick an edge which has a face.
	 *
	 * \note This is infact quite a simple check, mainly include this function so the intent is more obvious.
	 * We know these 2 verts will _always_ make up the loops edge
	 */
	void BGraph::BG_edge_ordered_verts_ex(BGEdge *edge, BGVert **r_v1, BGVert **r_v2,
		BGLoop *edge_loop)
	{
		assert(edge_loop->e == edge);
		*r_v1 = edge_loop->v;
		*r_v2 = edge_loop->next->v;
	}

	void BGraph::BG_edge_ordered_verts(BGEdge *edge, BGVert **r_v1, BGVert **r_v2)
	{
		BG_edge_ordered_verts_ex(edge, r_v1, r_v2, edge->l);
	}

	/**
	 * Returns the edge existing between v1 and v2, or NULL if there isn't one.
	 *
	 * \note multiple edges may exist between any two vertices, and therefore
	 * this function only returns the first one found.
	 */
	BGEdge *BGraph::BG_edge_exists(BGVert *v1, BGVert *v2)
	{
		return eofv_2(v1, v2);
	}

	/**
	 * Given a set of vertices \a varr, find out if
	 * all those vertices overlap an existing face.
	 *
	 * \note Making a face here is valid but in some cases you wont want to
	 * make a face thats part of another.
	 *
	 * \returns TRUE for overlap
	 *
	 */
	int BGraph::BG_face_exists_overlap(BGVert **varr, int len, BGFace **r_overlapface)
	{
		int i, amount;
		for (i = 0; i < len; i++)
		{
			BGIter viter;
			viter.init(varr[i]);
			BGFace *f;
			for (f = fofv_begin(viter); f != fofv_end(viter); f = fofv_next(viter))
			{
				amount = BG_verts_in_face(f, varr, len);
				if (amount >= len)
				{
					if (r_overlapface)
					{
						*r_overlapface = f;
					}
					return TRUE;
				}
			}
		}

		if (r_overlapface)
		{
			*r_overlapface = NULL;
		}

		return FALSE;
	}

	/**
	 * Given a set of vertices (varr), find out if
	 * there is a face with exactly those vertices
	 * (and only those vertices).
	 */
	int BGraph::BG_face_exists(BGVert **varr, int len, BGFace **r_existface)
	{
		int i, amount;

		for (i = 0; i < len; i++)
		{
			BGIter viter;
			viter.init(varr[i]);
			BGFace *f;
			for (f = fofv_begin(viter); f != fofv_end(viter); f = fofv_next(viter))
			{
				amount = BG_verts_in_face(f, varr, len);
				if (amount == len && amount == f->len)
				{
					if (r_existface)
					{
						*r_existface = f;
					}
					return TRUE;
				}
			}
		}

		if (r_existface)
		{
			*r_existface = NULL;
		}
		return FALSE;
	}

	/**
	 * Given a set of vertices and edges (\a varr, \a earr), find out if
	 * all those vertices are filled in by existing faces that _only_ use those vertices.
	 *
	 * This is for use in cases where creating a face is possible but would result in
	 * many overlapping faces.
	 *
	 * An example of how this is used: when 2 tri's are selected that share an edge,
	 * pressing Fkey would make a new overlapping quad (without a check like this)
	 *
	 * \a earr and \a varr can be in any order, however they _must_ form a closed loop.
	 */
	int BGraph::BG_face_exists_multi(BGVert **varr, BGEdge **earr, int len)
	{
		//BGFace *f;
		//BGEdge *e;
		//BGVert *v;
		//int ok;
		//int tot_tag;

		//BGIter fiter;
		//BGIter viter;

		//int i;

		//for (i = 0; i < len; i++) {
		//	/* save some time by looping over edge faces rather then vert faces
		//	 * will still loop over some faces twice but not as many */
		//	BG_ITER_ELEM (f, &fiter, earr[i], BG_FACES_OF_EDGE) {
		//		BG_elem_flag_disable(f, BG_ELEM_INTERNAL_TAG);
		//		BG_ITER_ELEM (v, &viter, f, BG_VERTS_OF_FACE) {
		//			BG_elem_flag_disable(v, BG_ELEM_INTERNAL_TAG);
		//		}
		//	}

		//	/* clear all edge tags */
		//	BG_ITER_ELEM (e, &fiter, varr[i], BG_EDGES_OF_VERT) {
		//		BG_elem_flag_disable(e, BG_ELEM_INTERNAL_TAG);
		//	}
		//}

		///* now tag all verts and edges in the boundary array as true so
		// * we can know if a face-vert is from our array */
		//for (i = 0; i < len; i++) {
		//	BG_elem_flag_enable(varr[i], BG_ELEM_INTERNAL_TAG);
		//	BG_elem_flag_enable(earr[i], BG_ELEM_INTERNAL_TAG);
		//}


		///* so! boundary is tagged, everything else cleared */


		///* 1) tag all faces connected to edges - if all their verts are boundary */
		//tot_tag = 0;
		//for (i = 0; i < len; i++) {
		//	BG_ITER_ELEM (f, &fiter, earr[i], BG_FACES_OF_EDGE) {
		//		if (!BG_elem_flag_test(f, BG_ELEM_INTERNAL_TAG)) {
		//			ok = TRUE;
		//			BG_ITER_ELEM (v, &viter, f, BG_VERTS_OF_FACE) {
		//				if (!BG_elem_flag_test(v, BG_ELEM_INTERNAL_TAG)) {
		//					ok = FALSE;
		//					break;
		//				}
		//			}

		//			if (ok) {
		//				/* we only use boundary verts */
		//				BG_elem_flag_enable(f, BG_ELEM_INTERNAL_TAG);
		//				tot_tag++;
		//			}
		//		}
		//		else {
		//			/* we already found! */
		//		}
		//	}
		//}

		//if (tot_tag == 0) {
		//	/* no faces use only boundary verts, quit early */
		//	return FALSE;
		//}

		///* 2) loop over non-boundary edges that use boundary verts,
		// *    check each have 2 tagges faces connected (faces that only use 'varr' verts) */
		//ok = TRUE;
		//for (i = 0; i < len; i++) {
		//	BG_ITER_ELEM (e, &fiter, varr[i], BG_EDGES_OF_VERT) {

		//		if (/* non-boundary edge */
		//		    BG_elem_flag_test(e, BG_ELEM_INTERNAL_TAG) == FALSE &&
		//		    /* ...using boundary verts */
		//		    BG_elem_flag_test(e->getStartPoint(), BG_ELEM_INTERNAL_TAG) == TRUE &&
		//		    BG_elem_flag_test(e->getEndPoint(), BG_ELEM_INTERNAL_TAG) == TRUE)
		//		{
		//			int tot_face_tag = 0;
		//			BG_ITER_ELEM (f, &fiter, e, BG_FACES_OF_EDGE) {
		//				if (BG_elem_flag_test(f, BG_ELEM_INTERNAL_TAG)) {
		//					tot_face_tag++;
		//				}
		//			}

		//			if (tot_face_tag != 2) {
		//				ok = FALSE;
		//				break;
		//			}

		//		}
		//	}

		//	if (ok == FALSE) {
		//		break;
		//	}
		//}

		//return ok;
		printf("%s: not implemented!\n", __func__);
		return 0;
	}

	/* same as 'BG_face_exists_multi' but built vert array from edges */
	int BGraph::BG_face_exists_multi_edge(BGEdge **earr, int len)
	{
		BGVert **varr;
		varr = new BGVert*[len];

		int ok;
		int i, i_next;

		ok = TRUE;
		for (i = len - 1, i_next = 0; i_next < len; (i = i_next++))
		{
			if (!(varr[i] = BG_edge_share_vert(earr[i], earr[i_next])))
			{
				ok = FALSE;
				break;
			}
		}

		if (ok == FALSE)
		{
			BGESH_ASSERT(0);
			free(varr);
			return FALSE;
		}

		ok = BG_face_exists_multi(varr, earr, len);
		delete varr;
		return ok;
	}

	/**
	 * low level function, only frees the vert,
	 * doesn't change or adjust surrounding geometry
	 */
	void BGraph::bm_kill_only_vert(BGraph *bm, BGVert *v)
	{
		bm->vpool.erase(v->getId());
	}

	/**
	 * low level function, only frees the edge,
	 * doesn't change or adjust surrounding geometry
	 */
	void BGraph::bm_kill_only_edge(BGraph *bm, BGEdge *e)
	{
		bm->epool.erase(e->getId());
	}

	/**
	 * low level function, only frees the face,
	 * doesn't change or adjust surrounding geometry
	 */
	void BGraph::bm_kill_only_face(BGraph *bm, BGFace *f)
	{
		bm->fpool.erase(f->getId());
	}

	/**
	 * low level function, only frees the loop,
	 * doesn't change or adjust surrounding geometry
	 */
	void BGraph::bm_kill_only_loop(BGraph *bm, BGLoop *l)
	{
		bm->lpool.erase(l->getId());
	}

	/**
	 * kills all edges associated with \a f, along with any other faces containing
	 * those edges
	 */
	void BGraph::BG_face_edges_kill(BGraph *bm, BGFace *f)
	{
		vector<BGEdge*> edges;
		edges.reserve(BG_NGON_STACK_SIZE);
		BGLoop *l_iter;
		BGLoop *l_first;

		l_iter = l_first = BG_FACE_FIRST_LOOP(f);
		do
		{
			edges.push_back(l_iter->e);
		} while ((l_iter = l_iter->next) != l_first);

		for (unsigned i = 0; i < edges.size(); i++)
		{
			BG_edge_kill(bm, edges[i]);
		}
	}

	/**
	 * kills all verts associated with \a f, along with any other faces containing
	 * those vertices
	 */
	void BGraph::BG_face_verts_kill(BGraph *bm, BGFace *f)
	{
		vector<BGVert*> verts;
		verts.reserve(BG_LOOPS_OF_FACE);
		BGLoop *l_iter;
		BGLoop *l_first;

		l_iter = l_first = BG_FACE_FIRST_LOOP(f);
		do
		{
			verts.push_back(l_iter->v);
		} while ((l_iter = l_iter->next) != l_first);

		for (unsigned i = 0; i < verts.size(); i++)
		{
			BG_vert_kill(bm, verts[i]);
		}
	}

	void BGraph::BG_face_kill(BGraph *bm, BGFace *f)
	{
		if (f->l_first)
		{
			BGLoop *l_iter, *l_next, *l_first;
			l_iter = l_first = f->l_first;
			do
			{
				l_next = l_iter->next;

				bmesh_radial_loop_remove(l_iter, l_iter->e);
				bm_kill_only_loop(bm, l_iter);

			} while ((l_iter = l_next) != l_first);
		}
		bm_kill_only_face(bm, f);
	}
	/**
	 * kills \a e and all faces that use it.
	 */
	void BGraph::BG_edge_kill(BGraph *bm, BGEdge *e)
	{

		bmesh_disk_edge_remove(e, (BGVert*)e->getStartPoint());
		bmesh_disk_edge_remove(e, (BGVert*)e->getEndPoint());

		if (e->l)
		{
			BGLoop *l = e->l, *lnext, *startl = e->l;

			do
			{
				lnext = l->radial_next;
				if (lnext->f == l->f)
				{
					BG_face_kill(bm, l->f);
					break;
				}

				BG_face_kill(bm, l->f);

				if (l == lnext)
					break;
				l = lnext;
			} while (l != startl);
		}

		bm_kill_only_edge(bm, e);
	}

	/**
	 * kills \a v and all edges that use it.
	 */
	void BGraph::BG_vert_kill(BGraph *bm, BGVert *v)
	{
		if (v->m_edge)
		{
			BGEdge *e, *nexte;

			e = v->m_edge;
			while (v->m_edge)
			{
				nexte = bmesh_disk_edge_next(e, v);
				BG_edge_kill(bm, e);
				e = nexte;
			}
		}

		bm_kill_only_vert(bm, v);
	}

	/**
	 * \brief Splice Vert
	 *
	 * Merges two verts into one (\a v into \a vtarget).
	 *
	 * \return Success
	 */
	int BGraph::BG_vert_splice(BGraph *bm, BGVert *v, BGVert *vtarget)
	{
		BGEdge *e;
		BGLoop *l;
		BGIter liter;

		/* verts already spliced */
		if (v == vtarget)
		{
			return FALSE;
		}

		/* retarget all the loops of v to vtarget */
		liter.init(v);
		for (l = lofv_begin(liter); l != lofv_end(liter); l = lofv_next(liter))
		{
			l->v = vtarget;
		}

		/* move all the edges from v's disk to vtarget's disk */
		while ((e = v->m_edge))
		{
			bmesh_disk_edge_remove(e, v);
			bmesh_edge_swapverts(e, v, vtarget);
			bmesh_disk_edge_append(e, vtarget);
		}

		/* v is unused now, and can be killed */
		BG_vert_kill(bm, v);

		return TRUE;
	}

	/**
	 * \brief Splice Edge
	 *
	 * Splice two unique edges which share the same two vertices into one edge.
	 *
	 * \return Success
	 *
	 * \note Edges must already have the same vertices.
	 */
	int BGraph::BG_edge_splice(BGraph *bm, BGEdge *e, BGEdge *etarget)
	{
		BGLoop *l;

		if (!BG_vert_in_edge(e, etarget->getStartPoint()) || !BG_vert_in_edge(e, etarget->getEndPoint()))
		{
			/* not the same vertices can't splice */
			return FALSE;
		}

		while (e->l)
		{
			l = e->l;
			assert(BG_vert_in_edge(etarget, l->v));
			assert(BG_vert_in_edge(etarget, l->next->v));
			bmesh_radial_loop_remove(l, e);
			bmesh_radial_append(etarget, l);
		}

		assert(bmesh_radial_length(e->l) == 0);

		/* removes from disks too */
		BG_edge_kill(bm, e);

		return TRUE;
	}
	//====================================================================================================


}//end namespace ldp


