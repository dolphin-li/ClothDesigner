#include "Algorithm/cloth/definations.h"
#include <QString>
#include <vector>
#include "Algorithm/tinyxml/tinyxml.h"
#include "Algorithm/tinyxml/tinystr.h"
#include <QDir>
#include <QFileInfo>

struct BatchSimulateManager
{
	enum BatchSimPhase{ INIT, SIM1, SIM2, ENDGAME };
	BatchSimulateManager()
	{
		m_poseRoot = "./data/Mocap/poses/";
		m_shapeXml = "./data/spring/sprint_femal.smpl.xml";
		m_maxBodyNum = 1000;
		m_timerIntervals = 5000;
		m_curPatternId = 0;
		m_maxShapeNum = 0;
		m_batchSimMode = ldp::BatchSimNotInit;
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
		m_shapeIter = m_shapeInd = 0;
	}
	void finish()
	{
		init();
		m_batchSimMode = ldp::BatchSimNotInit;
		m_patternXmls.clear();
		m_poseFiles.clear();
		m_maxShapeNum = 0;
		m_curPatternId = 0;
		m_shapeDoc.Clear();
		m_outputDoc.Clear();
	}
	QString m_saveRootPath;
	std::vector<int> m_shapeIndexes;
	QStringList m_patternXmls;
	QStringList m_poseFiles;
	QString m_poseRoot;
	QString m_shapeXml;
	QString m_posePath;
	TiXmlDocument m_outputDoc;
	TiXmlDocument m_shapeDoc;
	TiXmlElement* m_shapeElm;
	int m_shapeIter;
	int m_curPatternId;
	int m_shapeInd;
	int m_maxBodyNum;
	int m_maxShapeNum;
	int m_timerIntervals;
	ldp::BatchSimulateMode m_batchSimMode ;
	BatchSimPhase m_phase;

};