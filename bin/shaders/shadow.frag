varying vec4 fragment_position;

uniform sampler2D 	shadow_texture;
uniform mat4 		biased_MVP;

varying vec3 normal;
varying vec3 vpos;

float rand1(vec2 co)
{
   return fract(sin(dot(co.xy,vec2(12.9898,78.233))) * 43758.5453);
}

float rand2(vec2 co)
{
   return fract(sin(dot(co.xy,vec2(12.9898,68.233))) * 33058.5453);
}

void main()
{   
	/////////////shadow/////////////////////////////////////////////////////////////
	vec3 L = normalize(gl_LightSource[0].position.xyz - fragment_position);
	vec3 N = vec3(0, 0, 1);
	vec3 shadow_coord = biased_MVP * fragment_position;
	float visibility = 0.0;
	for(int i=-3; i<=3; i++)
	for(int j=-3; j<=3; j++)
	{
		vec2 pos=shadow_coord.xy+(vec2(i,j) + vec2(rand1(shadow_coord.xy), rand2(shadow_coord.xy))*1)*0.005;
		if(texture2D(shadow_texture, pos ).z  <  shadow_coord.z)
			visibility = visibility + 0.8;
		else visibility = visibility +1.0;
	}
	visibility=visibility/49.0;
	float shadowColor=dot(L, N)*visibility;//*0.8+0.1;

	///////////////////////////////////////phong///////////////////////////////////////////////
	vec3 n = normalize(normal);
	vec4 diffuse = vec4(0.0);
	vec4 specular = vec4(0.0);
	
	// ambient term
	vec4 ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
	
	// diffuse color
	vec4 kd = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
	
	// specular color
	vec4 ks = gl_FrontMaterial.specular * gl_LightSource[0].specular;
	
	// diffuse term
	vec3 lightDir = normalize(gl_LightSource[0].position.xyz - vpos);
	float NdotL = dot(n, lightDir);
	
	if (NdotL > 0.0)
		diffuse = kd * NdotL;
	
	// specular term
	vec3 rVector = normalize(2.0 * n * dot(n, lightDir) - lightDir);
	vec3 viewVector = normalize(-vpos);
	float RdotV = dot(rVector, viewVector);
	
	if (RdotV > 0.0)
		specular = ks * pow(RdotV, gl_FrontMaterial.shininess);

	gl_FragColor = ambient + diffuse*shadowColor + specular;
} 
