#version 130

out float Depth;

void main()
{
	gl_Position = ftransform();
	Depth = abs((gl_ModelViewMatrix * gl_Vertex).z);
}
