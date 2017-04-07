#include "GPUBuffers.h"

using namespace cml;

void GPUBuffers::SetShadowSize(int width, int height)
{
	m_ShadowWidth = width;
	m_ShadowHeight = height;
}

void GPUBuffers::SetFrameSize(int width, int height)
{
	m_FrameWidth = width;
	m_FrameHeight = height;
}

void GPUBuffers::ResetFrameSize(int width, int height)
{
	m_FrameWidth = width;
	m_FrameHeight = height;
	m_FrameColorTexture.BindTexture();
	m_FrameColorTexture.ImageTexture(4, GL_RGBA32F, m_FrameWidth, m_FrameHeight);
	m_FrameColorTexture.UnbindTexture();
	m_FrameDepthTexture.BindTexture();
	m_FrameDepthTexture.ImageTexture(4, GL_DEPTH_COMPONENT24, m_FrameWidth, m_FrameHeight);
	m_FrameDepthTexture.UnbindTexture();
}

GLFBO& GPUBuffers::GetShadowFBO()
{
	return m_ShadowFBO;
}

GLMSFBO& GPUBuffers::GetFrameFBO()
{
	return m_FrameFBO;
}

bool GPUBuffers::CreateTextures()
{
	if (!m_ShadowColorTexture.CreateRGBA32FTexture(m_ShadowWidth, m_ShadowHeight, GL_LINEAR) ||
		!m_ShadowDepthTexture.CreateDepthTexture(m_ShadowWidth, m_ShadowHeight, GL_LINEAR))
		return false;
	if (!m_FrameColorTexture.CreateRGBA32FTexture(m_FrameWidth, m_FrameHeight, 4) ||
		!m_FrameDepthTexture.CreateDepthTexture(m_FrameWidth, m_FrameHeight, 4))
		return false;
	return true;
}

void GPUBuffers::DeleteTextures()
{
	m_ShadowColorTexture.DeleteTexture();
	m_ShadowDepthTexture.DeleteTexture();
	m_FrameColorTexture.DeleteTexture();
	m_FrameDepthTexture.DeleteTexture();
}

bool GPUBuffers::CreateFBOs()
{
	m_ShadowFBO.CreateFBO();
	m_FrameFBO.CreateFBO();
	return true;
}

void GPUBuffers::DeleteFBOs()
{
	m_ShadowFBO.DeleteFBO();
	m_FrameFBO.DeleteFBO();
}

void GPUBuffers::AttachFBOTextures()
{
	m_ShadowFBO.AttachColorTexture(m_ShadowColorTexture);
	m_ShadowFBO.AttachDepthTexture(m_ShadowDepthTexture);
	m_FrameFBO.AttachColorTexture(m_FrameColorTexture);
	m_FrameFBO.AttachDepthTexture(m_FrameDepthTexture);
}
