//attribute vec4 vertex_position;     //In the local space
//attribute vec4 vertex_normal;       //In the local space


varying vec4 fragment_position;

varying vec3 normal;
varying vec3 vpos;

void main()
{ 
	//In the world space
	fragment_position = gl_Vertex; 	

	// vertex normal
	normal = normalize(gl_NormalMatrix * gl_Normal);
	
	// vertex position
	vpos = vec3(gl_ModelViewMatrix * gl_Vertex);
	
	// vertex position
	gl_Position = ftransform();
}
