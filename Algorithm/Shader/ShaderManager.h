#pragma once

#include "glsl.h"
#include <string>
class CShaderManager
{
public:
	enum ShaderType
	{
		none,
		phong,
		shaderEnd,
	};
	enum{ nShaders = shaderEnd};
public:
	CShaderManager();
	~CShaderManager();
	void create(const char* folder_of_shaders);
	void bind(ShaderType type);
	void unbind();
	void release();
	cwc::glShader* getCurShader();
	cwc::glShader* getShader(ShaderType type);
	ShaderType getCurType()const{return m_type;}
protected:
	void bind_phong();
	void bind_shadow();
private:
	cwc::glShader* m_shader[nShaders];
	cwc::glShaderObject* m_vshader[nShaders];
	cwc::glShaderObject* m_fshader[nShaders];
	std::string m_dir;
	ShaderType m_type;
};