#ifndef __LDP_BGRAPH_H__
#define __LDP_BGRAPH_H__
/**
 *  THIS IS A C++ WRAPPER OF BLENDER BGRAPH DATA STRUCTURE.
 *  BY: DONGPING LI
 *	\file blender/BGraph/BGraph.h
 *  \ingroup BGraph
 *
 * \addtogroup BGraph BGraph
 *
 * \brief BGraph is a non-manifold boundary representation designed to replace the current, limited EditMesh structure,
 * solving many of the design limitations and maintenance issues of EditMesh.
 */
#include <vector>
#include "ldpMat\ldp_basic_vec.h"
#include "GraphPoint.h"
#include "AbstractGraphCurve.h"
#include "GraphLoop.h"
#include <hash_map>
namespace ldp
{

	class BGraph;
	typedef GraphPoint BGVert;
	typedef AbstractGraphCurve BGEdge;
	class BGLoop;
	class BGFace;
	class BGIter;
	struct BGLI_mempool_chunk;
	struct BGLI_mempool_iter;
	struct BGLI_mempool;
	enum
	{
		BG_VERT = 1,
		BG_EDGE = 2,
		BG_LOOP = 4,
		BG_FACE = 8
	};

	class BGraph : public AbstractGraphObject
	{
	public:
		BGraph();
		virtual ~BGraph();

		virtual BGraph* clone()const;
		virtual Type getType()const { return TypeGraph; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		/**
		* Queries: Vert Of Mesh.
		* BGIter iter;for(BGVert* v = vofm_begin(iter); v != vofm_end(iter); v=vofm_next(iter)){...}
		* */
		BGVert* vofm_begin(BGIter& iter);
		BGVert* vofm_next(BGIter& iter);
		BGVert* vofm_end(BGIter& iter);
		const BGVert* vofm_begin(BGIter& iter)const;
		const BGVert* vofm_next(BGIter& iter)const;
		const BGVert* vofm_end(BGIter& iter)const;
		int vofm_count()const { return vpool.size(); }
		/**
		* Queries: Edge Of Mesh.
		* BGIter iter;for(BGEdge* e = eofm_begin(iter); e != eofm_end(iter); e=eofm_next(iter)){...}
		* */
		BGEdge* eofm_begin(BGIter& iter);
		BGEdge* eofm_next(BGIter& iter);
		BGEdge* eofm_end(BGIter& iter);
		const BGEdge* eofm_begin(BGIter& iter)const;
		const BGEdge* eofm_next(BGIter& iter)const;
		const BGEdge* eofm_end(BGIter& iter)const;
		int eofm_count()const { return epool.size(); }
		/**
		* Queries: Face Of Mesh
		* BGIter iter;for(BGFace* f = fofm_begin(iter); f != fofm_end(iter); f=fofm_next(iter)){...}
		* */
		BGFace* fofm_begin(BGIter& iter);
		BGFace* fofm_next(BGIter& iter);
		BGFace* fofm_end(BGIter& iter);
		const BGFace* fofm_begin(BGIter& iter)const;
		const BGFace* fofm_next(BGIter& iter)const;
		const BGFace* fofm_end(BGIter& iter)const;
		int fofm_count()const { return fpool.size(); }

		/**
		* Queries: Vert Of Face
		* Useage: BGIter iter;iter.init(f); for(BGVert* v = voff_begin(iter); v != voff_end(iter); v=voff_next(iter)){...}
		* */
		static BGVert* voff_begin(BGIter& iter);
		static BGVert* voff_next(BGIter& iter);
		static BGVert* voff_end(BGIter& iter);
		static int voff_count(const BGFace* f);
		/**
		* Queries: Edge Of Vertex
		* Useage: BGIter iter; iter.init(v); for(BGEdge* e = eofv_begin(iter); e != eofv_end(iter); e=eofv_next(iter)){...}
		* */
		static BGEdge* eofv_begin(BGIter& iter);
		static BGEdge* eofv_next(BGIter& iter);
		static BGEdge* eofv_end(BGIter& iter);
		static int eofv_count(BGVert* v);
		/**
		* Returns the edge existing between v1 and v2, or NULL if there isn't one.
		*
		* \note multiple edges may exist between any two vertices, and therefore
		* this function only returns the first one found.
		* */
		static BGEdge* eofv_2(BGVert* v1, BGVert* v2);
		/**
		* Queries: Edge Of Face
		* Useage: BGIter iter; iter.init(f); for(BGEdge* e = voff_begin(iter); e != voff_end(iter); e=voff_next(iter)){...}
		* */
		static BGEdge* eoff_begin(BGIter& iter);
		static BGEdge* eoff_next(BGIter& iter);
		static BGEdge* eoff_end(BGIter& iter);
		static int eoff_count(BGFace* f);
		/**
		* Queries: Face Of Vert
		* Useage: BGIter iter; iter.init(v); for(BGFace* f = fofv_begin(iter); f != fofv_end(iter); f= fofv_next(iter)){...}
		* */
		static BGFace* fofv_begin(BGIter& iter);
		static BGFace* fofv_next(BGIter& iter);
		static BGFace* fofv_end(BGIter& iter);
		static int fofv_count(BGVert* v);
		/**
		* Queries: Face Of Edge
		* Useage: BGIter iter; iter.init(e); for(BGFace* f = fofe_begin(iter); f != fofe_end(iter); f= fofe_next(iter)){...}
		* */
		static BGFace* fofe_begin(BGIter& iter);
		static BGFace* fofe_next(BGIter& iter);
		static BGFace* fofe_end(BGIter& iter);
		static int fofe_count(BGEdge* e);
		/**
		* Queries: Loop Of Vert
		* Useage: BGIter iter; iter.init(v); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
		* */
		static BGLoop* lofv_begin(BGIter& iter);
		static BGLoop* lofv_next(BGIter& iter);
		static BGLoop* lofv_end(BGIter& iter);
		static int lofv_count(BGVert* l);
		/**
		* Queries: Loop Of Edge
		* Useage: BGIter iter; iter.init(e); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
		* */
		static BGLoop* lofe_begin(BGIter& iter);
		static BGLoop* lofe_next(BGIter& iter);
		static BGLoop* lofe_end(BGIter& iter);
		static int lofe_count(BGEdge* e);
		/**
		* Queries: Loop Of Face
		* Useage: BGIter iter; iter.init(f); for(BGLoop* l = lofv_begin(iter); l != lofv_end(iter); l=lofv_next(iter)){...}
		* */
		static BGLoop* loff_begin(BGIter& iter);
		static BGLoop* loff_next(BGIter& iter);
		static BGLoop* loff_end(BGIter& iter);
		static int loff_count(BGFace* f);
		/**
		* Queries: Loops Of Loop
		* Useage: BGIter iter; iter.init(l); for(BGLoop* l2 = lofv_begin(iter); l2 != lofv_end(iter); l2=lofv_next(iter)){...}
		* */
		static BGLoop* lofl_begin(BGIter& iter);
		static BGLoop* lofl_next(BGIter& iter);
		static BGLoop* lofl_end(BGIter& iter);
		static int lofl_count(BGLoop* l);
	public:
		const static int BG_LOOP_RADIAL_MAX = 10000;
	public:
		//in bmesh_core.h
		BGVert*			create_vert(Float2 p);
		BGVert*			create_vert(const std::shared_ptr<BGVert>& rhs);
		BGEdge*			create_edge(const std::vector<std::shared_ptr<BGVert>>& vs);
		BGEdge*			create_edge(const std::vector<BGVert*>& vs);
		BGEdge*			create_edge(const std::shared_ptr<BGEdge>& e);
		BGFace*			create_face(BGVert **verts, BGEdge **edges, const int len);
		BGFace*			create_face(const std::vector<std::vector<std::shared_ptr<BGVert>>>& edges);
		BGFace*			create_face(const std::vector<std::shared_ptr<BGEdge>>& edges);
		BGFace*			create_face(const std::vector<BGEdge*>& edges);
		BGVert*			bmesh_semv(BGVert *tv, BGEdge *e, BGEdge **r_e);
		BGEdge*			bmesh_jekv(BGEdge *ke, BGVert *kv, const short check_edge_double);
		BGFace*			bmesh_sfme(BGFace *f, BGVert *v1, BGVert *v2, BGLoop **r_l);
		BGFace*			bmesh_jfke(BGFace *f1, BGFace *f2, BGEdge *e);
		BGFace*			BG_face_split(BGFace *f, BGVert *v1, BGVert *v2, BGLoop **r_l);
	private:
		BGFace*			bm_face_create_internal();
		BGLoop*			bm_loop_create(BGVert *v, BGEdge *e, BGFace *f);
		BGLoop*			bm_face_boundary_add(BGFace*f, BGVert* startv, BGEdge* starte);
		static void		bm_kill_only_vert(BGraph *bm, BGVert *v);
		static void		bm_kill_only_edge(BGraph *bm, BGEdge *e);
		static void		bm_kill_only_face(BGraph *bm, BGFace *f);
		static void		bm_kill_only_loop(BGraph *bm, BGLoop *l);
		static void		BG_face_edges_kill(BGraph *bm, BGFace *f);
		static void		BG_face_verts_kill(BGraph *bm, BGFace *f);
		static void		BG_face_kill(BGraph *bm, BGFace *f);
		static void		BG_edge_kill(BGraph *bm, BGEdge *e);
		static void		BG_vert_kill(BGraph *bm, BGVert *v);
		static int		bmesh_edge_separate(BGraph *bm, BGEdge *e, BGLoop *l_sep);//not implement yet..
		static int		BG_edge_splice(BGraph *bm, BGEdge *e, BGEdge *etarget);
		static int		BG_vert_splice(BGraph *bm, BGVert *v, BGVert *vtarget);
		static int		bmesh_vert_separate(BGraph *bm, BGVert *v, BGVert ***r_vout, int *r_vout_len);//not implement yet..
		static int		bmesh_loop_reverse(BGraph *bm, BGFace *f);
		int				BG_vert_separate(BGraph *bm, BGVert *v, BGVert ***r_vout, int *r_vout_len, BGEdge **e_in, int e_in_len);
		//in bmesh_struct.h
		static bool		bmesh_disk_edge_append(BGEdge* e, BGVert* v);
		static BGEdge*	bmesh_disk_edge_next(const BGEdge *e, const BGVert *v);
		static BGEdge*	bmesh_disk_edge_prev(const BGEdge *e, const BGVert *v);
		static BGEdge*	bmesh_disk_edge_exists(const BGVert *v1, const BGVert *v2);
		static void		bmesh_disk_edge_remove(BGEdge *e, BGVert *v);
		static int		bmesh_disk_facevert_count(const BGVert *v);
		static BGEdge*	bmesh_disk_faceedge_find_first(BGEdge *e, const BGVert *v);
		static BGEdge*	bmesh_disk_faceedge_find_next(BGEdge *e, const BGVert *v);
		static int		bmesh_disk_validate(int len, const BGEdge *e, const BGVert *v);
		static int		bmesh_disk_count(const BGVert *v);
		static int		bmesh_radial_validate(int radlen, const BGLoop *l);
		static void		bmesh_radial_append(BGEdge* e, BGLoop* l);
		static int		bmesh_radial_length(const BGLoop *l);
		static void		bmesh_radial_loop_remove(BGLoop *l, BGEdge *e);
		static BGLoop*	bmesh_radial_faceloop_find_first(BGLoop *l, const BGVert *v);
		static BGLoop*	bmesh_radial_faceloop_find_next(BGLoop *l, const BGVert *v);
		static int		bmesh_radial_face_find(const BGEdge *e, const BGFace *f);
		static int		bmesh_radial_facevert_count(const BGLoop *l, const BGVert *v);
		static int		bmesh_vert_in_edge(const BGEdge *e, const BGVert *v);
		static int		bmesh_verts_in_edge(const BGVert *v1, const BGVert *v2, const BGEdge *e);
		static BGVert*	bmesh_edge_other_vert_get(BGEdge *e, const BGVert *v);
		static int		bmesh_edge_swapverts(BGEdge *e, BGVert *orig, BGVert *newv);
		static int		bmesh_loop_validate(const BGFace *f);
		static int		bm_loop_length(BGLoop* l);
		//in bmesh_queries.h
		static int		BG_vert_in_face(BGFace *f, BGVert *v);
		static int		BG_verts_in_face(BGFace *f, BGVert **varr, int len);
		static int		BG_edge_in_face(BGFace *f, BGEdge *e);
		static int		BG_vert_in_edge(const BGEdge *e, const BGVert *v);
		static int		BG_verts_in_edge(BGVert *v1, BGVert *v2, BGEdge *e);
		static float	BG_edge_calc_length(BGEdge *e);
		static int		BG_edge_face_pair(BGEdge *e, BGFace **r_fa, BGFace **r_fb);
		static int		BG_edge_loop_pair(BGEdge *e, BGLoop **r_la, BGLoop **r_lb);
		static BGVert *	BG_edge_other_vert(BGEdge *e, BGVert *v);
		static BGLoop *	BG_face_other_edge_loop(BGFace *f, BGEdge *e, BGVert *v);
		static BGLoop *	BG_face_other_vert_loop(BGFace *f, BGVert *v_prev, BGVert *v);
		static BGLoop *	BG_loop_other_vert_loop(BGLoop *l, BGVert *v);
		static int		BG_vert_edge_count_nonwire(BGVert *v);
		static int		BG_vert_edge_count(BGVert *v);
		static int		BG_edge_face_count(BGEdge *e);
		static int		BG_vert_face_count(BGVert *v);
		static BGEdge *	BG_vert_other_disk_edge(BGVert *v, BGEdge *e);
		static int		BG_vert_is_wire(BGVert *v);
		static int		BG_edge_is_wire(BGEdge *e);
		static int		BG_vert_is_manifold(BGVert *v);
		static int		BG_edge_is_manifold(BGEdge *e);
		static int		BG_edge_is_boundary(BGEdge *e);
		static BGEdge *	BG_edge_exists(BGVert *v1, BGVert *v2);
		static int		BG_face_exists_overlap(BGVert **varr, int len, BGFace **r_existface);
		static int		BG_face_exists(BGVert **varr, int len, BGFace **r_existface);
		static int		BG_face_exists_multi(BGVert **varr, BGEdge **earr, int len);
		static int		BG_face_exists_multi_edge(BGEdge **earr, int len);
		static int		BG_face_share_edge_count(BGFace *f1, BGFace *f2);
		static int		BG_edge_share_face_count(BGEdge *e1, BGEdge *e2);
		static int		BG_edge_share_vert_count(BGEdge *e1, BGEdge *e2);
		static BGVert *	BG_edge_share_vert(BGEdge *e1, BGEdge *e2);
		static BGLoop *	BG_face_vert_share_loop(BGFace *f, BGVert *v);
		static BGLoop *	BG_face_edge_share_loop(BGFace *f, BGEdge *e);
		static void		BG_edge_ordered_verts(BGEdge *edge, BGVert **r_v1, BGVert **r_v2);
		static void		BG_edge_ordered_verts_ex(BGEdge *edge, BGVert **r_v1, BGVert **r_v2,
			BGLoop *edge_loop);
	private:
		/*element pools*/
		std::hash_map<size_t, std::shared_ptr<BGVert>> vpool;
		std::hash_map<size_t, std::shared_ptr<BGFace>> fpool;
		std::hash_map<size_t, std::shared_ptr<BGEdge>> epool;
		std::hash_map<size_t, std::shared_ptr<BGLoop>> lpool;
		mutable std::hash_map<AbstractGraphObject*, AbstractGraphObject*> m_objMapAfterClone;
	};

	class BGLoop : public AbstractGraphObject
	{
		friend class BGIter;
		friend class BGraph;
	public:
		virtual BGLoop* clone()const { throw std::exception("BGLoop::clone(): cannot be called"); }
		virtual Type getType()const { return TypeGraphLoop; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const { return nullptr; }
		virtual void fromXML(TiXmlElement* self) {}
	private:
		/* circular linked list of loops which all use the same edge as this one '->e',
		 * but not necessarily the same vertex (can be either v1 or v2 of our own '->e') */
		BGLoop *radial_next = nullptr, *radial_prev = nullptr;

		/* these were originally commented as private but are used all over the code */
		/* can't use ListBase API, due to head */
		BGLoop *next = nullptr, *prev = nullptr; /* next/prev verts around the face */
		BGVert *v;
		BGEdge *e; /* edge, using verts (v, next->v) */
		BGFace *f;
	};

	class BGFace : public AbstractGraphObject
	{
		friend class BGIter;
		friend class BGraph;
	public:
		virtual AbstractGraphObject* clone()const
		{
			BGFace* loop = (BGFace*)create(getType());
			loop->setSelected(isSelected());
			loop->len = len;
			loop->l_first = l_first;
			return loop;

		}
		virtual Type getType()const { return TypeGraphLoop; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const { return nullptr; }
		virtual void fromXML(TiXmlElement* self) {}
	private:
		int len = 0; /*includes all boundary loops*/
		BGLoop *l_first = nullptr;
	};



#pragma region --macros

#define BGRAPH_ALL_FACES(f, iter, mesh)\
	BGIter iter; \
	for (BGFace* f = (mesh).fofm_begin(iter); f != (mesh).fofm_end(iter); f = (mesh).fofm_next(iter))

#define BGRAPH_ALL_FACES_CONST(f, iter, mesh)\
	BGIter iter; \
	for (const BGFace* f = (mesh).fofm_begin(iter); f != (mesh).fofm_end(iter); f = (mesh).fofm_next(iter))

#define BGRAPH_ALL_VERTS(v, iter, mesh)\
	BGIter iter; \
	for (BGVert* v = (mesh).vofm_begin(iter); v != (mesh).vofm_end(iter); v = (mesh).vofm_next(iter))

#define BGRAPH_ALL_VERTS_CONST(v, iter, mesh)\
	BGIter iter; \
	for (const BGVert* v = (mesh).vofm_begin(iter); v != (mesh).vofm_end(iter); v = (mesh).vofm_next(iter))

#define BGRAPH_ALL_EDGES(e, iter, mesh)\
	BGIter iter; \
	for (BGEdge* e = (mesh).eofm_begin(iter); e != (mesh).eofm_end(iter); e = (mesh).eofm_next(iter))

#define BGRAPH_ALL_EDGES_CONST(e, iter, mesh)\
	BGIter iter; \
	for (const BGEdge* e = (mesh).eofm_begin(iter); e != (mesh).eofm_end(iter); e = (mesh).eofm_next(iter))

#define BGRAPH_V_OF_F(v, f, iter, mesh)\
	BGIter iter; \
	iter.init(f); \
	for (BGVert* v = (mesh).voff_begin(iter); v != (mesh).voff_end(iter); v = (mesh).voff_next(iter))

#define BGRAPH_F_OF_V(f, v, iter, mesh)\
	BGIter iter; \
	iter.init(v); \
	for (BGFace* f = (mesh).fofv_begin(iter); f != (mesh).fofv_end(iter); f = (mesh).fofv_next(iter))

#define BGRAPH_E_OF_V(e, v, iter, mesh)\
	BGIter iter; \
	iter.init(v); \
	for (BGEdge* e = (mesh).eofv_begin(iter); e != (mesh).eofv_end(iter); e = (mesh).eofv_next(iter))

#define BGRAPH_F_OF_E(f, e, iter, mesh)\
	BGIter iter; \
	iter.init(e); \
	for (BGFace* f = (mesh).fofe_begin(iter); f != (mesh).fofe_end(iter); f = (mesh).fofe_next(iter))

#define BGRAPH_E_OF_F(e, f, iter, mesh)\
	BGIter iter; \
	iter.init(f); \
	for (BGEdge* e = (mesh).eoff_begin(iter); e != (mesh).eoff_end(iter); e = (mesh).eoff_next(iter))

#pragma endregion
	//=============================================================================================

	//Iterators
	class BGIter
	{
		friend class BGraph;
	public:
		void clear();
		void init(BGLoop* l);
		void init(const BGVert* v);
		void init(const BGFace* f);
		void init(const BGEdge* e);
	private:
		int count = 0;
		BGVert *firstvert = nullptr, *nextvert = nullptr;
		BGEdge *firstedge = nullptr, *nextedge = nullptr;
		BGLoop *firstloop = nullptr, *nextloop = nullptr, *ldata = nullptr, *l = nullptr;
		BGFace *firstpoly = nullptr, *nextpoly = nullptr;
		const BGVert* c_vdata = nullptr;
		const BGEdge* c_edata = nullptr;
		const BGFace* c_pdata = nullptr;
		std::hash_map<size_t, std::shared_ptr<BGVert>>::iterator viter;
		std::hash_map<size_t, std::shared_ptr<BGFace>>::iterator fiter;
		std::hash_map<size_t, std::shared_ptr<BGEdge>>::iterator eiter;
		std::hash_map<size_t, std::shared_ptr<BGLoop>>::iterator liter;
		std::hash_map<size_t, std::shared_ptr<BGVert>>::const_iterator cviter;
		std::hash_map<size_t, std::shared_ptr<BGFace>>::const_iterator cfiter;
		std::hash_map<size_t, std::shared_ptr<BGEdge>>::const_iterator ceiter;
		std::hash_map<size_t, std::shared_ptr<BGLoop>>::const_iterator cliter;
	};
};//namespace ldp


#endif//__LDP_BGRAPH_H__