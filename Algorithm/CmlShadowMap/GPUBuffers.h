#pragma once

#include "GLHelper.h"

class GPUBuffers
{
public:

	void SetShadowSize(int width, int height);
	void SetFrameSize(int width, int height);
	void ResetFrameSize(int width, int height);

	cml::GLFBO& GetShadowFBO();
	cml::GLMSFBO& GetFrameFBO();

	bool CreateTextures();
	void DeleteTextures();

	bool CreateFBOs();
	void DeleteFBOs();
	void AttachFBOTextures();

private:

	int m_ShadowWidth;
	int m_ShadowHeight;

	int m_FrameWidth;
	int m_FrameHeight;

	cml::GLFBO m_ShadowFBO;
	cml::GLTexture m_ShadowColorTexture;
	cml::GLTexture m_ShadowDepthTexture;

	cml::GLMSFBO m_FrameFBO;
	cml::GLMSTexture m_FrameColorTexture;
	cml::GLMSTexture m_FrameDepthTexture;
};
