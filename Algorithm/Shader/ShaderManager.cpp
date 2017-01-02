#include "ShaderManager.h"
CShaderManager::CShaderManager()
{
	for(int i=0; i<nShaders; i++)
	{
		m_shader[i] = NULL;
		m_vshader[i] = NULL;
		m_fshader[i] = NULL;
	}
	m_type = none;
}

CShaderManager::~CShaderManager()
{
	
}

void CShaderManager::release()
{
	for(int i=0; i<nShaders; i++)
	{
		if(m_vshader[i]) delete m_vshader[i];
		if(m_fshader[i]) delete m_fshader[i];
		if(m_shader[i]) delete m_shader[i];
	}
}

inline std::string fullfile(std::string path, std::string name)
{
	if (path == "")
		return name;
	if (path.back() != '/' && path.back() != '\\')
		path.append("/");
	return path + name;
}

void CShaderManager::create(const char* folder_of_shaders)
{
	release();

	printf("%s\n", gluErrorString(glGetError()));

	const static std::string names[nShaders] ={
		"none", "phong", "shadow"
	};

	for(int i=1; i<nShaders; i++)
	{
		std::string fname = fullfile(folder_of_shaders, names[i]+".frag");
		std::string vname = fullfile(folder_of_shaders, names[i]+".vert");
		m_vshader[i] = new cwc::aVertexShader();
		m_fshader[i] = new cwc::aFragmentShader();
		if(m_fshader[i]->load(fname.c_str())|| m_vshader[i]->load(vname.c_str()))
		{
			printf("cound not load shader: %s \n or \n %s\n", fname.c_str(), vname.c_str());
			delete m_vshader[i];
			delete m_fshader[i];
			m_vshader[i] = NULL;
			m_fshader[i] = NULL;
			continue;
		}
		m_fshader[i]->compile();
		m_vshader[i]->compile();
		
		m_shader[i] = new cwc::glShader();
		m_shader[i]->addShader(m_fshader[i]);
		m_shader[i]->addShader(m_vshader[i]);
		m_shader[i]->link();
		printf("[shader log %d]: %s\n", i, m_shader[i]->getLinkerLog());
	}// i	
}

void CShaderManager::bind(ShaderType type)
{
	m_type = type;
	if(!m_shader[m_type])
		return;
	m_shader[m_type]->begin();
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	switch(m_type)
	{
	default:
		printf("error: non-supported shader type");
		return;
	case phong:
		bind_phong();
		break;
	case shadow:
		bind_shadow();
		break;
	}
}

void CShaderManager::unbind()
{
	if(!m_shader[m_type])
		return;
	glPopAttrib();
	m_shader[m_type]->end();
}

void CShaderManager::bind_phong()
{
	float light_amb[] = { 0.05, 0.05, 0.05, 1 };
	float light_dif[] = { 1, 1, 1, 1 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
}

void CShaderManager::bind_shadow()
{
	float light_amb[] = { 0.05, 0.05, 0.05, 1 };
	float light_dif[] = { 1, 1, 1, 1 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
}

cwc::glShader* CShaderManager::getCurShader()
{
	return m_shader[m_type];
}

cwc::glShader* CShaderManager::getShader(ShaderType type)
{
	return m_shader[type];
}
