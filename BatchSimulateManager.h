#include "Algorithm/cloth/definations.h"
#include <QString>
#include <vector>
#include "Algorithm/tinyxml/tinyxml.h"
#include "Algorithm/tinyxml/tinystr.h"
#include <QDir>
#include <QFileInfo>


struct BatchSimulateManager
{
	//enum BatchSimPhase{NO_INIT,INIT_SIM,PRECOMPUTE,LOAD_BODY,OUTPUT};
	enum BatchSimPhase{ INIT, SIM1, SIM2, ENDGAME };
	BatchSimulateManager()
	{
		m_poseRoot = "./data/Mocap/poses/";
		m_shapeXml = "./data/spring/sprint_femal.smpl.xml";
		m_maxBodyNum = 5;
		init();
	}
	void recordPoseFiles()
	{
		QDir poseDir(m_poseRoot);
		poseDir.setNameFilters(QStringList("*.xml"));
		m_poseFiles = poseDir.entryList();
	}
	void init()
	{
		m_shapeElm = nullptr;
		m_phase = BatchSimPhase::INIT;
		m_batchSimMode = ldp::BatchSimNotInit;
		m_shapeIter = m_shapeInd = 0;
		
	}
	QString m_saveRootPath;
	std::vector<int> m_shapeIndexes;
	QString m_poseRoot;
	QString m_shapeXml;
	QString m_posePath;
	TiXmlDocument m_outputDoc;
	TiXmlDocument m_shapeDoc;
	TiXmlElement* m_shapeElm;
	QStringList m_poseFiles;
	int m_shapeIter;
	int m_shapeInd;
	int m_maxBodyNum;
	ldp::BatchSimulateMode m_batchSimMode = ldp::BatchSimNotInit;
	BatchSimPhase m_phase;

};