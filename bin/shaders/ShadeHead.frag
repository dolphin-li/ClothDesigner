#version 130

uniform vec3 LightColor1;
uniform vec3 LightColor2;
uniform vec3 LightColor3;
uniform vec3 LightColor4;

uniform vec3 LightDir1;
uniform vec3 LightDir2;
uniform vec3 LightDir3;
uniform vec3 LightDir4;

uniform sampler2D ScalpColor;
uniform sampler2D DepthMap;

uniform float Ambient;
uniform float Diffuse;
uniform float Specular;
uniform vec3 Color;
uniform float Shadow;

in vec4 LightViewPosition1;
in vec4 LightViewPosition2;
in vec4 LightViewPosition3;
in vec4 LightViewPosition4;

in float LightViewDepth1;
in float LightViewDepth2;
in float LightViewDepth3;
in float LightViewDepth4;

in vec3 Normal;
in vec3 View;

in vec2 TexCoord;

void GetShading(out vec3 val1, out vec3 val2, out vec3 val3, out vec3 val4)
{
	float Shininess = 10.;
	//vec3 texColor = texture2D(ScalpColor, TexCoord).rgb;
	vec3 diffuseColor = Diffuse * Color;
	vec3 specularColor = Specular * vec3(1., 1., 1.);
	vec3 n = normalize(Normal);
	vec3 e = normalize(View);

	// 1

	float nlDot1 = dot(LightDir1, n);
	if (nlDot1 < 0.)
		val1 = vec3(0,0,0);//-nlDot1 * diffuseColor + pow(max(dot(reflect(-LightDir1, -n), e), 0.), Shininess) * specularColor;
	else
		val1 = nlDot1 * diffuseColor + pow(max(dot(reflect(-LightDir1, n), e), 0.), Shininess) * specularColor;

	// 2

	float nlDot2 = dot(LightDir2, n);
	if (nlDot2 < 0.)
		val2 = vec3(0,0,0);// = -nlDot2 * diffuseColor + pow(max(dot(reflect(-LightDir2, -n), e), 0.), Shininess) * specularColor;
	else
		val2 = nlDot2 * diffuseColor + pow(max(dot(reflect(-LightDir2, n), e), 0.), Shininess) * specularColor;

	// 3

	float nlDot3 = dot(LightDir3, n);
	if (nlDot3 < 0.)
		val3 = vec3(0,0,0);// = -nlDot3 * diffuseColor + pow(max(dot(reflect(-LightDir3, -n), e), 0.), Shininess) * specularColor;
	else
		val3 = nlDot3 * diffuseColor + pow(max(dot(reflect(-LightDir3, n), e), 0.), Shininess) * specularColor;

	// 4

	float nlDot4 = dot(LightDir4, n);
	if (nlDot4 < 0.)
		val4 = vec3(0,0,0);// = -nlDot4 * diffuseColor + pow(max(dot(reflect(-LightDir4, -n), e), 0.), Shininess) * specularColor;
	else
		val4 = nlDot4 * diffuseColor + pow(max(dot(reflect(-LightDir4, n), e), 0.), Shininess) * specularColor;
}

void GetOpacityFactor(out float val1, out float val2, out float val3, out float val4)
{
	vec3 n = normalize(Normal);
	float angleBias = 0.033;
	float baseBias = 0.022;

	// 1

	vec4 lpos1 = LightViewPosition1 / LightViewPosition1.w;
	vec2 texPos1 = (lpos1.xy * 0.5 + vec2(0.5, 0.5)) * 0.5;

	val1 = 1.;
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 4; ++j){
			float scalpDepth = texture2D(DepthMap, texPos1 + vec2((float(i) - 1.5) / 1024., (float(j) - 1.5) / 1024.)).z;
			if (LightViewDepth1 > scalpDepth + angleBias * (tan(acos(dot(n, LightDir1)))) + baseBias)
				val1 -= 1. / 16.;
		}
	}
	val1 = 1. - (1. - val1) * Shadow;

	// 2

	vec4 lpos2 = LightViewPosition2 / LightViewPosition2.w;
	vec2 texPos2 = (lpos2.xy * 0.5 + vec2(0.5, 0.5)) * 0.5 + vec2(0.5, 0.);

	val2 = 1.;
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 4; ++j){
			float scalpDepth = texture2D(DepthMap, texPos2 + vec2((float(i) - 1.5) / 1024., (float(j) - 1.5) / 1024.)).z;
			if (LightViewDepth2 > scalpDepth + angleBias * (tan(acos(dot(n, LightDir2)))) + baseBias)
				val2 -= 1. / 16.;
		}
	}
	val2 = 1. - (1. - val2) * Shadow;

	// 3

	vec4 lpos3 = LightViewPosition3 / LightViewPosition3.w;
	vec2 texPos3 = (lpos3.xy * 0.5 + vec2(0.5, 0.5)) * 0.5 + vec2(0., 0.5);

	val3 = 1.;
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 4; ++j){
			float scalpDepth = texture2D(DepthMap, texPos3 + vec2((float(i) - 1.5) / 1024., (float(j) - 1.5) / 1024.)).z;
			if (LightViewDepth3 > scalpDepth + angleBias * (tan(acos(dot(n, LightDir3)))) + baseBias)
				val3 -= 1. / 16.;
		}
	}
	val3 = 1. - (1. - val3) * Shadow;

	// 4

	vec4 lpos4 = LightViewPosition4 / LightViewPosition4.w;
	vec2 texPos4 = (lpos4.xy * 0.5 + vec2(0.5, 0.5)) * 0.5 + vec2(0.5, 0.5);

	val4 = 1.;
	for (int i = 0; i < 4; ++i){
		for (int j = 0; j < 4; ++j){
			float scalpDepth = texture2D(DepthMap, texPos4 + vec2((float(i) - 1.5) / 1024., (float(j) - 1.5) / 1024.)).z;
			if (LightViewDepth4 > scalpDepth + angleBias * (tan(acos(dot(n, LightDir4)))) + baseBias)
				val4 -= 1. / 16.;
		}
	}
	val4 = 1. - (1. - val4) * Shadow;
}

void main()
{
	vec3 shading1, shading2, shading3, shading4;
	GetShading(shading1, shading2, shading3, shading4);

	vec3 ambient = vec3(Ambient, Ambient, Ambient);

	float opacityFactor1, opacityFactor2, opacityFactor3, opacityFactor4;
	GetOpacityFactor(opacityFactor1, opacityFactor2, opacityFactor3, opacityFactor4);

	gl_FragColor = vec4((
		LightColor1 * shading1 * opacityFactor1 +
		LightColor2 * shading2 * opacityFactor2 +
		LightColor3 * shading3 * opacityFactor3 +
		LightColor4 * shading4 * opacityFactor4 +
		ambient), 1.);
}
