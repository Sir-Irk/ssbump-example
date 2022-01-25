#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 uv;

out vec2 UV;
out vec3 ViewDirTangentSpace;
out vec3 LightDirTangentSpace;

uniform mat4 mvp;
uniform mat4 model;
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform float uvScale;

vec3 transform_basis(vec3 v, vec3 t, vec3 b, vec3 n) {
    return vec3(dot(t, v), dot(b, v), dot(n, v));
}

void main() {

    gl_Position = mvp * vec4(position, 1.0f);

    vec3 worldPos = vec3(model * vec4(position, 1.0f));
    vec3 lightDirWorldSpace = lightPos - worldPos;
    vec3 viewDirWorldSpace  = viewPos - worldPos;

    mat3 mod = transpose(mat3(model));
    vec3 basis1 = vec3(mod[0][0], mod[0][1], mod[0][2]);
    vec3 basis2 = vec3(mod[1][0], mod[1][1], mod[1][2]);
    vec3 basis3 = vec3(mod[2][0], mod[2][1], mod[2][2]);

    //vec3 tan = vec3(1.0f, 0.0f, 0.0f);
    //vec3 bi  = vec3(0.0f, 1.0f, 0.0f);
    //vec3 nor = vec3(0.0f, 0.0f, -1.0f);

    //vec3 t = normalize(transform_basis(tan, basis1, basis2, basis3));
    //vec3 n = normalize(transform_basis(nor, basis1, basis2, basis3));
    //vec3 b = normalize(transform_basis(bi,  basis1, basis2, basis3));
    vec3 t = normalize(transform_basis(tangent, basis1, basis2, basis3));
    vec3 n = normalize(transform_basis(normal, basis1, basis2, basis3));
    vec3 b = (cross(n, -t));

    LightDirTangentSpace = transform_basis(lightDirWorldSpace, t, b, n);
    ViewDirTangentSpace  = transform_basis(viewDirWorldSpace, t, b, n);

    UV = uv * uvScale;
}
