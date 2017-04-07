#version 130

uniform vec3 ViewPoint;

uniform mat4 LightModelViewMatrix1;
uniform mat4 LightModelViewMatrix2;
uniform mat4 LightModelViewMatrix3;
uniform mat4 LightModelViewMatrix4;

uniform mat4 LightModelViewProjMatrix1;
uniform mat4 LightModelViewProjMatrix2;
uniform mat4 LightModelViewProjMatrix3;
uniform mat4 LightModelViewProjMatrix4;

out vec4 LightViewPosition1;
out vec4 LightViewPosition2;
out vec4 LightViewPosition3;
out vec4 LightViewPosition4;

out float LightViewDepth1;
out float LightViewDepth2;
out float LightViewDepth3;
out float LightViewDepth4;

out vec3 Normal;
out vec3 View;

out vec2 TexCoord;

void main()
{
	LightViewDepth1 = abs((LightModelViewMatrix1 * gl_Vertex).z);
	LightViewDepth2 = abs((LightModelViewMatrix2 * gl_Vertex).z);
	LightViewDepth3 = abs((LightModelViewMatrix3 * gl_Vertex).z);
	LightViewDepth4 = abs((LightModelViewMatrix4 * gl_Vertex).z);

	LightViewPosition1 = LightModelViewProjMatrix1 * gl_Vertex;
	LightViewPosition2 = LightModelViewProjMatrix2 * gl_Vertex;
	LightViewPosition3 = LightModelViewProjMatrix3 * gl_Vertex;
	LightViewPosition4 = LightModelViewProjMatrix4 * gl_Vertex;

	View = normalize(ViewPoint - gl_Vertex.xyz);
	Normal = normalize(gl_Normal);

	TexCoord = gl_MultiTexCoord0.xy;

	gl_Position = ftransform();
}
