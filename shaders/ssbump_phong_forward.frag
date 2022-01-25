#version 430 core

layout(location = 0) out vec4 fragColor;

in vec2 UV;
in vec3 LightDirTangentSpace;
in vec3 ViewDirTangentSpace;
in vec3 NormalInTangentSpace;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;

uniform float timer;
uniform bool normalMapping;

const float pi = 3.14159265;

const vec3 bumpBasis[3] = vec3[](
    vec3(sqrt(2.0) / sqrt(3.0), 0.0, 1.0 / sqrt(3.0)),
    vec3(-1.0 / sqrt(6.0), 1.0 / sqrt(2.0), 1.0 / sqrt(3.0)),
    vec3(-1.0 / sqrt(6.0), -1.0 / sqrt(2.0), 1.0 / sqrt(3.0))
);

float saturate(float x){
    return max(0.0f, min(1.0f, x));
}

void main() {

    vec3 normal = normalize(texture(normalMap, UV*2.0f).rgb * 2.0f + 1.0f);
    normal = normalize((bumpBasis[0] * normal.x + bumpBasis[1] * normal.y + bumpBasis[2] * normal.z));
    normal.y = -normal.y;

    vec3 dp;
    dp.x = saturate(dot(normal, bumpBasis[0]));
    dp.y = saturate(dot(normal, bumpBasis[1]));
    dp.z = saturate(dot(normal, bumpBasis[2]));
    dp *= dp;
    float sum = dot(dp, vec3(1.0f, 1.0f, 1.0f));


    float diff = max(dot(normal, normalize(LightDirTangentSpace)), 0.0);
    vec3 halfwayDir = normalize(normalize(LightDirTangentSpace) + normalize(ViewDirTangentSpace));
    float energyConserv = (8.0 + 100.0) / (8.0 * pi);
    float spec = energyConserv * pow(max(dot(normal, halfwayDir), 0.0), 100.0);

    float dist = length(LightDirTangentSpace);
    float attenuation = 1.0f / (dist * dist);

    vec3 lightColor = vec3(1.0f, 1.0f, 0.9f); 

    float bright = 0.1f;
	vec3 albedo   = texture(diffuseMap, UV*2.0f).rgb;

    vec3 ambient  = vec3(0.01f * albedo);
    //vec3 diffuse  = lightColor * albedo * diff * bright * attenuation;
    vec3 diffuse  = lightColor * diff * bright * attenuation;
    //vec3 specular = lightColor * texture(mat.specular, UV).rgb * bright * spec * attenuation;
    vec3 specular = lightColor * 0.01f * bright * spec * attenuation;
    //vec3 specular = vec3(0.0f);

    //fragColor = vec4(ambient + (diffuse/sum/pi), 1.0f);
    fragColor = vec4((diffuse/sum/pi), 1.0f);
    //fragColor = vec4(ambient + (diffuse/pi) + specular, 1.0f);
 //   fragColor = vec4(ambient, 1.0f);
    //fragColor = vec4(albedo, 1.0f);
}
