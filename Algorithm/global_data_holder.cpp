#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "cloth\LevelSet3D.h"
GlobalDataHolder g_dataholder;

void GlobalDataHolder::init()
{
	m_clothManager.reset(new ldp::ClothManager);

	debug_2();
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
	ldp::LevelSet3D lvt;
	lvt.load("data/mannequin.set");
	lvt.save("data/mannequin1.set");
	return;

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
	m_clothManager->addClothPiece(std::shared_ptr<ldp::ClothPiece>(piece));

	// debug create levelset
	const float step = 0.003;
	ldp::Float3 range = body->boundingBox[1] - body->boundingBox[0];
	ldp::Float3 start = body->boundingBox[0] - ldp::Float3(0, 0, 0.12f)*range;
	ldp::Float3 end = body->boundingBox[1] + ldp::Float3(0, 0, 0.12f)*range;
	ldp::Int3 res = (end - start) / step;
	start = ldp::Float3(-0.169504836, 0.789619565, -0.134123757);
	res = ldp::Int3(110, 236, 84);
	ldp::LevelSet3D lv;
	lv.create(res, start, step);
	lv.fromMesh(*body);
	lv.save("data/mannequin.set");
}