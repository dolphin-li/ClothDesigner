#pragma once

#include <stdio.h>

#include <gl/glew.h>
#include <QtOpenGL>

namespace cml
{
	template<GLenum ShaderType>
	class GLShader
	{
	public:

		GLShader()
		{
			m_shaderID = 0;
			m_fileName[0] = '\0';
		}

		bool CreateShader()
		{
			m_shaderID = QGLFunctions(QGLContext::currentContext()).glCreateShader(ShaderType);
			return m_shaderID != 0;
		}

		bool LoadShaderFromFile(const char* filename)
		{
			FILE* shaderFile = fopen(filename, "rb");
			fseek(shaderFile, 0, SEEK_END);
			int shaderSize = ftell(shaderFile);
			rewind(shaderFile);
			char* shaderSource = new char[shaderSize + 1];
			fread(shaderSource, sizeof(char), shaderSize, shaderFile);
			fclose(shaderFile);
			shaderSource[shaderSize] = '\0';
			memcpy(m_fileName, filename, 260);
			QGLFunctions(QGLContext::currentContext()).glShaderSource(m_shaderID, 1, (const char**)&shaderSource, NULL);
			return true;
		}

		bool LoadShaderFromString(const char* shaderSource)
		{
			m_fileName[0] = '\0';
			QGLFunctions(QGLContext::currentContext()).glShaderSource(m_shaderID, 1, &shaderSource, NULL);
			return true;
		}

		bool CompileShader()
		{
			QGLFunctions(QGLContext::currentContext()).glCompileShader(m_shaderID);
			GLint compileStatus;
			QGLFunctions(QGLContext::currentContext()).glGetShaderiv(m_shaderID, GL_COMPILE_STATUS, &compileStatus);
			if (!compileStatus){
				GLint infoSize;
				QGLFunctions(QGLContext::currentContext()).glGetShaderiv(m_shaderID, GL_INFO_LOG_LENGTH, &infoSize);
				GLchar* infoSource = new char[infoSize];
				GLint readSize = 0;
				QGLFunctions(QGLContext::currentContext()).glGetShaderInfoLog(m_shaderID, infoSize, &readSize, infoSource);
				printf("Shader compile error\n");
				printf("File name: %s\n", m_fileName);
				printf("InfoLog:\n%s\n", infoSource);
				delete infoSource;
				return false;
			}
			return true;
		}

		bool DeleteShader()
		{
			QGLFunctions(QGLContext::currentContext()).glDeleteShader(m_shaderID);
			m_shaderID = 0;
			return true;
		}

		bool CreateShaderFromFile(const char* filename)
		{
			if (!CreateShader())
				return false;
			if (!LoadShaderFromFile(filename) || !CompileShader()){
				DeleteShader();
				return false;
			}
			return true;
		}

		bool CreateShaderFromString(const char* shaderSource)
		{
			if (!CreateShader())
				return false;
			if (!LoadShaderFromString(shaderSource) || !CompileShader()){
				DeleteShader();
				return false;
			}
			return true;
		}

		GLuint GetShader() const
		{
			return m_shaderID;
		}

	private:

		GLuint m_shaderID;
		char m_fileName[260];
	};

	typedef GLShader<GL_VERTEX_SHADER> GLVertShader;
	typedef GLShader<GL_FRAGMENT_SHADER> GLFragShader;

	class GLProgram
	{
	public:

		GLProgram()
		{
			m_programID = 0;
		}

		bool CreateProgram()
		{
			m_programID = QGLFunctions(QGLContext::currentContext()).glCreateProgram();
			return true;
		}

		bool DeleteProgram()
		{
			QGLFunctions(QGLContext::currentContext()).glDeleteProgram(m_programID);
			m_programID = 0;
			return true;
		}

		bool AttachVertexShader(GLVertShader* vertexShader)
		{
			QGLFunctions(QGLContext::currentContext()).glAttachShader(m_programID, vertexShader->GetShader());
			return true;
		}

		bool AttachFragmentShader(GLFragShader* fragmentShader)
		{
			QGLFunctions(QGLContext::currentContext()).glAttachShader(m_programID, fragmentShader->GetShader());
			return true;
		}

		bool DetachVertexShader(GLVertShader* vertexShader)
		{
			QGLFunctions(QGLContext::currentContext()).glDetachShader(m_programID, vertexShader->GetShader());
			return true;
		}

		bool DetachFragmentShader(GLFragShader* fragmentShader)
		{
			QGLFunctions(QGLContext::currentContext()).glDetachShader(m_programID, fragmentShader->GetShader());
			return true;
		}

		bool LinkProgram()
		{
			QGLFunctions(QGLContext::currentContext()).glLinkProgram(m_programID);
			GLint infoSize;
			QGLFunctions(QGLContext::currentContext()).glGetProgramiv(m_programID, GL_INFO_LOG_LENGTH, &infoSize);
			GLchar* infoSource = new char[infoSize];
			GLint readSize = 0;
			QGLFunctions(QGLContext::currentContext()).glGetProgramInfoLog(m_programID, infoSize, &readSize, infoSource);
			GLint linkStatue;
			QGLFunctions(QGLContext::currentContext()).glGetProgramiv(m_programID, GL_LINK_STATUS, &linkStatue);
			if (!linkStatue){
				printf("GLProgram::LinkProgram - failed to link shader.\n");
				printf("InfoLog: \n%s\n", infoSource);
				delete infoSource;
				return false;
			}
			delete infoSource;
			return true;
		}

		bool UseProgram() const
		{
			QGLFunctions(QGLContext::currentContext()).glUseProgram(m_programID);
			return true;
		}

		bool DisuseProgram() const
		{
			QGLFunctions(QGLContext::currentContext()).glUseProgram(0);
			return true;
		}

		bool SetUniform1f(const char* valueName, float value0) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniform1f(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), value0);
			return true;
		}

		bool SetUniform2f(const char* valueName, float value0, float value1) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniform2f(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), value0, value1);
			return true;
		}

		bool SetUniform3f(const char* valueName, float value0, float value1, float value2) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniform3f(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), value0, value1, value2);
			return true;
		}

		bool SetUniform4f(const char* valueName, float value0, float value1, float value2, float value3) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniform4f(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), value0, value1, value2, value3);
			return true;
		}

		bool SetUniform1i(const char* valueName, int value0) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniform1i(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), value0);
			return true;
		}

		bool SetUniformMatrix4fv(const char* valueName, const float* data) const
		{
			QGLFunctions(QGLContext::currentContext()).glUniformMatrix4fv(QGLFunctions(QGLContext::currentContext()).glGetUniformLocation(m_programID, valueName), 1, false, data);
			return true;
		}

		GLuint GetProgram()
		{
			return m_programID;
		}

		bool CreateProgramFromShaders(GLVertShader* vertexShader, GLFragShader* fragmentShader)
		{
			if (!CreateProgram())
				return false;
			if (!AttachVertexShader(vertexShader) || !AttachFragmentShader(fragmentShader) || !LinkProgram()){
				DeleteProgram();
				return false;
			}
			return true;
		}

		bool CreateProgramFromShaders(GLVertShader* vertexShader, GLFragShader* fragmentShader, const char* ourName)
		{
			if (!CreateProgram())
				return false;
			if (!AttachVertexShader(vertexShader) || !AttachFragmentShader(fragmentShader)){
				DeleteProgram();
				return false;
			}
			glBindFragDataLocation(m_programID, 0, ourName);
			if (!LinkProgram()){
				DeleteProgram();
				return false;
			}
			return true;
		}

		bool CreateProgramFromFiles(const char* vertexName, const char* fragmentName)
		{
			if (!CreateProgram())
				return false;
			GLVertShader vertexShader;
			GLFragShader fragmentShader;
			if (!vertexShader.CreateShaderFromFile(vertexName) || !fragmentShader.CreateShaderFromFile(fragmentName)){
				DeleteProgram();
				return false;
			}
			bool shaderStatus = CreateProgramFromShaders(&vertexShader, &fragmentShader);
			vertexShader.DeleteShader();
			fragmentShader.DeleteShader();
			if (!shaderStatus)
				DeleteProgram();
			return shaderStatus;
		}

		bool CreateProgramFromFiles(const char* vertexName, const char* fragmentName, const char* outName)
		{
			if (!CreateProgram())
				return false;
			GLVertShader vertexShader;
			GLFragShader fragmentShader;
			if (!vertexShader.CreateShaderFromFile(vertexName) || !fragmentShader.CreateShaderFromFile(fragmentName)){
				DeleteProgram();
				return false;
			}
			bool shaderStatus = CreateProgramFromShaders(&vertexShader, &fragmentShader, outName);
			vertexShader.DeleteShader();
			fragmentShader.DeleteShader();
			if (!shaderStatus)
				DeleteProgram();
			return shaderStatus;
		}

		bool CreateProgramFromStrings(const char* vertexSource, const char* fragmentSource)
		{
			if (!CreateProgram())
				return false;
			GLVertShader vertexShader;
			GLFragShader fragmentShader;
			if (!vertexShader.CreateShaderFromString(vertexSource) || !fragmentShader.CreateShaderFromString(fragmentSource)){
				DeleteProgram();
				return false;
			}
			bool shaderStatus = CreateProgramFromShaders(&vertexShader, &fragmentShader);
			vertexShader.DeleteShader();
			fragmentShader.DeleteShader();
			if (!shaderStatus)
				DeleteProgram();
			return shaderStatus;
		}

		bool CreateProgramFromStrings(const char* vertexSource, const char* fragmentSource, const char* outName)
		{
			if (!CreateProgram())
				return false;
			GLVertShader vertexShader;
			GLFragShader fragmentShader;
			if (!vertexShader.CreateShaderFromString(vertexSource) || !fragmentShader.CreateShaderFromString(fragmentSource)){
				DeleteProgram();
				return false;
			}
			bool shaderStatus = CreateProgramFromShaders(&vertexShader, &fragmentShader, outName);
			vertexShader.DeleteShader();
			fragmentShader.DeleteShader();
			if (!shaderStatus)
				DeleteProgram();
			return shaderStatus;
		}

	private:
		GLuint m_programID;
	};

	template<GLenum VBOType>
	class GLVBO
	{
	public:

		GLVBO()
		{
			m_VBOID = 0;
		}

		bool CreateVBO()
		{
			QGLFunctions(QGLContext::currentContext()).glGenBuffers(1, &m_VBOID);
			return true;
		}

		bool FillVBO(const void* data, int dataSize, bool dynamic)
		{
			if (!BindVBO())
				return false;
			QGLFunctions(QGLContext::currentContext()).glBufferData(VBOType, dataSize, data, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
			QGLFunctions(QGLContext::currentContext()).glBindBuffer(VBOType, 0);
			return true;
		}

		bool BindVBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindBuffer(VBOType, m_VBOID);
			return true;
		}

		bool UnbindVBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindBuffer(VBOType, 0);
			return true;
		}

		bool ReplaceVBO(GLuint VBOID)
		{
			m_VBOID = VBOID;
			return true;
		}

		bool DeleteVBO()
		{
			QGLFunctions(QGLContext::currentContext()).glDeleteBuffers(1, &m_VBOID);
			m_VBOID = 0;
			return true;
		}

		GLuint GetVBO() const
		{
			return m_VBOID;
		}

	private:

		GLuint m_VBOID;
	};

	typedef GLVBO<GL_ARRAY_BUFFER> GLVertexVBO;
	typedef GLVBO<GL_ELEMENT_ARRAY_BUFFER> GLIndexVBO;

	class GLTexture
	{
	public:

		GLTexture()
		{
			m_textureID = 0;
		}

		bool CreateTexture()
		{
			glGenTextures(1, &m_textureID);
			return true;
		}

		bool DeleteTexture()
		{
			glDeleteTextures(1, &m_textureID);
			m_textureID = 0;
			return true;
		}

		GLuint GetTexture()
		{
			return m_textureID;
		}

		void GetTextureSize(int& width, int& height) const
		{
			width = m_textureWidth;
			height = m_textureHeight;
		}

		bool ActiveTexture(int unitID) const
		{
			QGLFunctions(QGLContext::currentContext()).glActiveTexture(GL_TEXTURE0 + unitID);
			return true;
		}

		bool BindTexture() const
		{
			glBindTexture(GL_TEXTURE_2D, m_textureID);
			return true;
		}

		bool UnbindTexture() const
		{
			glBindTexture(GL_TEXTURE_2D, 0);
			return true;
		}

		bool BindTextureUnit(int unitID) const
		{
			ActiveTexture(unitID);
			BindTexture();
			return true;
		}

		bool UnbindTextureUnit(int unitID) const
		{
			ActiveTexture(unitID);
			UnbindTexture();
			return true;
		}

		bool ImageTexture(GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum imageFormat, GLenum dataType, const GLvoid* data)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			glTexImage2D(GL_TEXTURE_2D, level, internalFormat, width, height, border, imageFormat,dataType, data);
			return true;
		}

		bool SubImageTexture(GLint level, GLint offsetX, GLint offsetY, GLint width, GLint height, GLenum imageFormat, GLenum dataType, const GLvoid* data)
		{
			glTexSubImage2D(GL_TEXTURE_2D, level, offsetX, offsetY, width, height, imageFormat, dataType, data);
			return true;
		}

		bool CreateDepthTexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
			UnbindTexture();
			return true;
		}

		bool CreateRTexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, NULL);
			UnbindTexture();
			return true;
		}

		bool CreateR32FTexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL);
			UnbindTexture();
			return true;
		}

		bool CreateRGBATexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
			UnbindTexture();
			return true;
		}

		bool CreateRGBA32FTexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT,NULL);
			UnbindTexture();
			return true;
		}

		bool CreateRGBA32UITexture(int width, int height, int filter)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			ImageTexture(0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, NULL);
			UnbindTexture();
			return true;
		}

	private:

		GLuint m_textureID;
		int m_textureWidth;
		int m_textureHeight;
	};

	class GLMSTexture
	{
	public:

		GLMSTexture()
		{
			m_textureID = 0;
		}

		bool CreateTexture()
		{
			glGenTextures(1, &m_textureID);
			return true;
		}

		bool DeleteTexture()
		{
			glDeleteTextures(1, &m_textureID);
			m_textureID = 0;
			return true;
		}

		GLuint GetTexture() const
		{
			return m_textureID;
		}

		void GetTextureSize(int& width, int& height) const
		{
			width = m_textureWidth;
			height = m_textureHeight;
		}

		bool ActiveTexture(int unitID) const
		{
			QGLFunctions(QGLContext::currentContext()).glActiveTexture(GL_TEXTURE0 + unitID);
			return true;
		}

		bool BindTexture() const
		{
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_textureID);
			return true;
		}

		bool UnbindTexture() const
		{
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
			return true;
		}

		bool BindTextureUnit(int unitID) const
		{
			ActiveTexture(unitID);
			BindTexture();
			return true;
		}

		bool UnbindTextureUnit(int unitID) const
		{
			ActiveTexture(unitID);
			UnbindTexture();
			return true;
		}

		bool ImageTexture(GLsizei sampleNum, GLint internalFormat, GLsizei width, GLsizei height)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, sampleNum, internalFormat, width, height, GL_TRUE);
			return true;
		}

		bool CreateDepthTexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_DEPTH_COMPONENT24, width, height);
			UnbindTexture();
			return true;
		}

		bool CreateRTexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_RED, width, height);
			UnbindTexture();
			return true;
		}

		bool CreateR32FTexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_R32F, width, height);
			UnbindTexture();
			return true;
		}

		bool CreateRGBATexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_RGBA, width, height);
			UnbindTexture();
			return true;
		}

		bool CreateRGBA32FTexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_RGBA32F, width, height);
			UnbindTexture();
			return true;
		}

		bool CreateRGBA32UITexture(int width, int height, int sampleNum)
		{
			m_textureWidth = width;
			m_textureHeight = height;
			CreateTexture();
			BindTexture();
			ImageTexture(sampleNum, GL_RGBA32UI, width, height);
			UnbindTexture();
			return true;
		}

	private:

		GLuint m_textureID;
		int m_textureWidth;
		int m_textureHeight;
	};

	class GLFBO
	{
	public:

		GLFBO()
		{
			m_FBOID = 0;
			m_colorTexture = NULL;
			m_depthTexture = NULL;
		}

		bool CreateFBO()
		{
			QGLFunctions(QGLContext::currentContext()).glGenFramebuffers(1, &m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			return m_FBOID != 0;
		}

		bool DeleteFBO()
		{
			QGLFunctions(QGLContext::currentContext()).glDeleteFramebuffers(1, &m_FBOID);
			m_FBOID = 0;
			return true;
		}

		GLuint GetFBO()
		{
			return m_FBOID;
		}

		GLTexture& GetColorTexture()
		{
			return *m_colorTexture;
		}

		GLTexture& GetDepthTexture()
		{
			return *m_depthTexture;
		}

		bool GetTextureSize(int& width, int& height)
		{
			if (m_colorTexture)
				m_colorTexture->GetTextureSize(width, height);
			else if (m_depthTexture)
				m_depthTexture->GetTextureSize(width, height);
			else
				return false;
			return true;
		}

		bool BindFBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			return true;
		}

		bool UnbindFBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return true;
		}

		bool AttachColorTexture(GLTexture& colorTexture)
		{
			m_colorTexture = &colorTexture;
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture.GetTexture(), 0);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return true;
		}

		bool AttachDepthTexture(GLTexture& depthTexture)
		{
			m_depthTexture = &depthTexture;
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture.GetTexture(), 0);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return true;
		}

	private:

		GLuint m_FBOID;
		GLTexture* m_colorTexture;
		GLTexture* m_depthTexture;
	};

	class GLMSFBO
	{
	public:

		GLMSFBO()
		{
			m_FBOID = 0;
			m_colorTexture = NULL;
			m_depthTexture = NULL;
		}

		bool CreateFBO()
		{
			QGLFunctions(QGLContext::currentContext()).glGenFramebuffers(1, &m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			return m_FBOID != 0;
		}

		void DeleteFBO()
		{
			QGLFunctions(QGLContext::currentContext()).glDeleteFramebuffers(1, &m_FBOID);
			m_FBOID = 0;
		}

		void BindFBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
		}
		void UnbindFBO() const
		{
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}

		GLuint GetFBO()
		{
			return m_FBOID;
		}

		GLMSTexture& GetColorTexture()
		{
			return *m_colorTexture;
		}

		GLMSTexture& GetDepthTexture()
		{
			return *m_depthTexture;
		}

		bool GetTextureSize(int& width, int& height)
		{
			if (m_colorTexture)
				m_colorTexture->GetTextureSize(width, height);
			else if (m_depthTexture)
				m_depthTexture->GetTextureSize(width, height);
			else
				return false;
			return true;
		}

		bool AttachColorTexture(GLMSTexture& colorTexture)
		{
			m_colorTexture = &colorTexture;
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, colorTexture.GetTexture(), 0);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return true;
		}

		bool AttachDepthTexture(GLMSTexture& depthTexture)
		{
			m_depthTexture = &depthTexture;
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, m_FBOID);
			QGLFunctions(QGLContext::currentContext()).glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depthTexture.GetTexture(), 0);
			QGLFunctions(QGLContext::currentContext()).glBindFramebuffer(GL_FRAMEBUFFER, 0);
			return true;
		}

	private:

		GLuint m_FBOID;
		GLMSTexture* m_colorTexture;
		GLMSTexture* m_depthTexture;
	};
}
