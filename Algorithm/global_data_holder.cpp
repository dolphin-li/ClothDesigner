#include "global_data_holder.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
GlobalDataHolder g_dataholder;

void GlobalDataHolder::init()
{
	m_clothManager.reset(new ldp::ClothManager);
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