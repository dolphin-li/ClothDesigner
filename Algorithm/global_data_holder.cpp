#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "cloth\LevelSet3D.h"
#include "cloth\HistoryStack.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
GlobalDataHolder g_dataholder;

void GlobalDataHolder::init()
{
	m_clothManager.reset(new ldp::ClothManager);
	m_historyStack.reset(new ldp::HistoryStack);
	m_historyStack->init(m_clothManager.get());
}

void GlobalDataHolder::debug_1()
{
	m_clothManager->bodyMesh()->loadObj("data/1_avt.obj", true, false);

	auto body = m_clothManager->bodyMesh();
	ldp::Float3 bodyCt = (body->boundingBox[0] + body->boundingBox[1])*0.5f;
	ldp::Mat3f R = ldp::QuaternionF().fromAngles(ldp::Float3(ldp::PI_S / 2, 0, 0)).toRotationMatrix3();
	body->rotateBy(R, bodyCt);
	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;

	auto piece = new ldp::ClothPiece();
	piece->mesh3d().loadObj("data/1_pt.obj", true, false);
	piece->mesh3d().rotateBy(R, bodyCt);
	piece->mesh3d().scaleByCenter(1.2);
	piece->mesh3d().translate(ldp::Float3(0, 0, -60));
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));
}

void GlobalDataHolder::debug_2()
{
	m_clothManager->bodyMesh()->loadObj("data/mannequin.obj", true, false);

	auto body = m_clothManager->bodyMesh();
	body->scaleBy(0.00075, 0);
	body->translate(ldp::Float3(0.36, 0.6, 0.02));
	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;

	auto piece = new ldp::ClothPiece();
	piece->mesh3d().loadObj("data/drs_mod.obj", true, false);
	piece->mesh3d().scaleBy(ldp::Float3(0.001, 0.0009, 0.0009), 0);
	piece->mesh3dInit().cloneFrom(&piece->mesh3d());
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));
	// debug create levelset
	try
	{
		m_clothManager->bodyLevelSet()->load("data/mannequin.set");
	} catch (std::exception e)
	{
		const float step = 0.003;
		ldp::Float3 start = ldp::Float3(-0.169504836, 0.789619565, -0.134123757);
		ldp::Int3 res = ldp::Int3(110, 236, 84);
		m_clothManager->bodyLevelSet()->create(res, start, step);
		m_clothManager->bodyLevelSet()->fromMesh(*body);
		m_clothManager->bodyLevelSet()->save("data/mannequin.set");
	}
}

void GlobalDataHolder::debug_3()
{
	m_clothManager->bodyMesh()->loadObj("data/mannequin_scaled.obj", true, false);

	auto body = m_clothManager->bodyMesh();
	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;

	auto piece = new ldp::ClothPiece();
	piece->mesh3d().loadObj("data/drs_scaled.obj", true, false);
	piece->mesh3dInit().cloneFrom(&piece->mesh3d());
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));

	// debug create levelset
	try
	{
		m_clothManager->bodyLevelSet()->load("data/mannequin_scaled.set");
	} catch (std::exception e)
	{
		const float step = 0.003;
		auto bmin = body->boundingBox[0];
		auto bmax = body->boundingBox[1];
		auto brag = bmax - bmin;
		bmin -= 0.1f * brag;
		bmax += 0.1f * brag;
		ldp::Int3 res = (bmax - bmin) / step;
		ldp::Float3 start = bmin;
		m_clothManager->bodyLevelSet()->create(res, start, step);
		m_clothManager->bodyLevelSet()->fromMesh(*body);
		m_clothManager->bodyLevelSet()->save("data/mannequin_scaled.set");
	}
}

void GlobalDataHolder::debug_4()
{
	m_clothManager->clear();
	m_clothManager->bodyMesh()->loadObj("data/debug4_body.obj", true, false);

	auto body = m_clothManager->bodyMesh();
	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;

	auto piece = new ldp::ClothPiece();
	piece->mesh3d().loadObj("data/debug4_piece1.obj", true, false);
	piece->mesh3dInit().cloneFrom(&piece->mesh3d());
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));
	piece = new ldp::ClothPiece();
	piece->mesh3d().loadObj("data/debug4_piece2.obj", true, false);
	piece->mesh3dInit().cloneFrom(&piece->mesh3d());
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));

	// in this example, the first 12 verts should be stithed
	for (int k = 0; k < 12; k++)
	{
		if (k == 5 || k == 11) continue;
		ldp::StitchPoint a, b;
		a.vids = ldp::Int2(k, k );
		a.w = 0.3;
		b.vids = ldp::Int2(k, k + 1);
		b.w = 0.7;
		m_clothManager->addStitchVert(m_clothManager->clothPiece(0), a, m_clothManager->clothPiece(1), b);
	}

	// debug create levelset
	try
	{
		m_clothManager->bodyLevelSet()->load("data/debug4_body.set");
	} catch (std::exception e)
	{
		const float step = 0.003;
		auto bmin = body->boundingBox[0];
		auto bmax = body->boundingBox[1];
		auto brag = bmax - bmin;
		bmin -= 0.1f * brag;
		bmax += 0.1f * brag;
		ldp::Int3 res = (bmax - bmin) / step;
		ldp::Float3 start = bmin;
		m_clothManager->bodyLevelSet()->create(res, start, step);
		m_clothManager->bodyLevelSet()->fromMesh(*body);
		m_clothManager->bodyLevelSet()->save("data/debug4_body.set");
	}
}

void GlobalDataHolder::debug_5()
{
	m_clothManager->clear();
	m_clothManager->loadPiecesFromSvg("data/Basic_Blouse_10_2016.svg");
	m_clothManager->bodyMeshInit()->loadObj("data/wm2_15k.obj", true, false);

	auto body = m_clothManager->bodyMeshInit();
	ldp::TransformInfo info;
	info.setIdentity();
	info.translate((body->boundingBox[0] + body->boundingBox[1])*-0.5f);
	info.rotate(ldp::QuaternionF().fromAngles(ldp::Float3(ldp::PI_S / 2, 0, ldp::PI_S)).toRotationMatrix3(), 0);
	info.scale(2.619848, 0);
	m_clothManager->setBodyMeshTransform(info);

	auto mat = body->default_material;
	mat.diff = ldp::Float3(0.5, 0.7, 0.8);
	body->material_list.clear();
	body->material_list.push_back(mat);
	for (auto& f : body->face_list)
		f.material_index = 0;
}


