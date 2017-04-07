#version 130

#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_texture_multisample : enable

uniform sampler2DMS HeadMap;

out vec4 OutColor;

void main()
{
	ivec2 texSize = textureSize(HeadMap);
	ivec2 texPos = ivec2(gl_TexCoord[0].xy * texSize);
	OutColor = (
		texelFetch(HeadMap, texPos, 0) +
		texelFetch(HeadMap, texPos, 1) +
		texelFetch(HeadMap, texPos, 2) +
		texelFetch(HeadMap, texPos, 3)) * 0.25;
}
