#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "glad.c"
#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include "si_math.h"
#include "types.h"

#define SI_MEMORY_IMPLEMENTATION
#include "si_memory.h"

#define SI_NORMALMAP_STATIC
#define SI_NORMALMAP_IMPLEMENTATION
#include "si_normalmap.h"

#include "opengl_helper.c"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct program_memory {
    si_primary_buffer buffer;
    si_memory_arena arena;
};

typedef struct vertex_data {
    si_v3 p;
    si_v3 n;
    si_v3 t;
    si_v2 uv;
} vertex_data;

static void
key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main(void)
{
    struct program_memory mem = { 0 };
    mem.buffer = si_allocate_primary_buffer(si_megabytes(16), 0);
    si_initialize_arena(&mem.arena, mem.buffer.size, mem.buffer.data);

    if (!glfwInit()) {
        printf("Failed to initialize glfw\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1280, 1024, "ssbump demo", NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwSetErrorCallback(error_callback);
    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(opengl_debug_callback, NULL);

    glfwSwapInterval(1);

    stbi_set_flip_vertically_on_load(true);

    i32 w, h, c;
    u32* diffuseImg = (u32*)stbi_load("textures/broken_tiles_01.tga", &w, &h, &c, 4);
    assert(diffuseImg);
    GLuint diffuse = create_texture(diffuseImg, w, h, false);

    u32* ssbumpImg = (u32*)stbi_load("textures/ssbump.png", &w, &h, NULL, 4);
    // u32 *ssbumpImg  = (u32 *)stbi_load("textures/face-ssbump.png", &w, &h, NULL, 4);
    assert(ssbumpImg);
    u32* normalImg = sinm_normal_map(ssbumpImg, w, h, 80.0f, 2.0f, sinm_greyscale_average, false);
    GLuint ssbump = create_texture(ssbumpImg, w, h, true);
    GLuint normal = create_texture(normalImg, w, h, true);

    // clang-format off
    vertex_data quad[4] = {
		{.p = { 0.5f,  0.5f, 0.0f}, .n = {0.0f, 0.0f, -1.0f}, .t = {1.0f, 0.0f, 0.0f}, .uv = {1.0f, 1.0f}},  // top right
		{.p = { 0.5f, -0.5f, 0.0f}, .n = {0.0f, 0.0f, -1.0f}, .t = {1.0f, 0.0f, 0.0f}, .uv = {1.0f, 0.0f}},  // bottom right
		{.p = {-0.5f,  0.5f, 0.0f}, .n = {0.0f, 0.0f, -1.0f}, .t = {1.0f, 0.0f, 0.0f}, .uv = {0.0f, 1.0f}},  // top left
		{.p = {-0.5f, -0.5f, 0.0f}, .n = {0.0f, 0.0f, -1.0f}, .t = {1.0f, 0.0f, 0.0f}, .uv = {0.0f, 0.0f}},  // bottom left
	};
    // clang-format on

    GLuint vbo, vao;
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(quad[0]), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(quad[0]), (void*)(3 * sizeof(f32)));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(quad[0]), (void*)(6 * sizeof(f32)));
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(quad[0]), (void*)(9 * sizeof(f32)));

    GLuint shader = create_program_from_files("shaders/ssbump_phong_forward.vert", "shaders/ssbump_phong_forward.frag", &mem.arena);
    glUseProgram(shader);

    STBI_FREE(diffuseImg);
    STBI_FREE(ssbumpImg);

    GLint mvpUni = get_uniform_location(shader, "mvp");
    GLint modelUni = get_uniform_location(shader, "model");
    GLint viewPosUni = get_uniform_location(shader, "viewPos");
    GLint lightPosUni = get_uniform_location(shader, "lightPos");
    GLint uvScaleUni = get_uniform_location(shader, "uvScale");
    GLint diffuseUni = get_uniform_location(shader, "diffuseMap");
    GLint normalUni = get_uniform_location(shader, "normalMap");

    glUniform1i(diffuseUni, 0);
    glUniform1i(normalUni, 1);

    glEnable(GL_FRAMEBUFFER_SRGB);

    f32 yRadians = 0.2f;
    si_v3 lightPos = { 0.1f, 0.1, -0.5f };
    while (!glfwWindowShouldClose(window)) {

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            shader = create_program_from_files("shaders/ssbump_phong_forward.vert", "shaders/ssbump_phong_forward.frag", &mem.arena);
            glUseProgram(shader);
            mvpUni = get_uniform_location(shader, "mvp");
            modelUni = get_uniform_location(shader, "model");
            viewPosUni = get_uniform_location(shader, "viewPos");
            lightPosUni = get_uniform_location(shader, "lightPos");
            uvScaleUni = get_uniform_location(shader, "uvScale");
            glUniform1i(diffuseUni, 0);
            glUniform1i(normalUni, 1);
        }

        i32 winW, winH;
        glfwGetFramebufferSize(window, &winW, &winH);
        f32 ratio = winW / (float)winH;

        // Rotate the plane's model on the Y axis
        f32 rotSpeed = 0.005f;
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            yRadians += rotSpeed;
        } else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            yRadians -= rotSpeed;
        }

        si_mat4x4 m = si_mat4x4_rot(si_mat4x4_identity(), 0.5f, new_si_v3(1.0f, 0.0f, 0.0f));
        m = si_mat4x4_rot(m, yRadians, new_si_v3(0.0f, 1.0f, 0.0f));

        si_v3 viewPos = { 0.0f, 0.0f, -1.5f };
        si_mat4x4 v = si_mat4x4_translate(si_mat4x4_identity(), viewPos);
        si_mat4x4 p = si_perspective(60, ratio, 0.0f, 100.0f);
        si_mat4x4 mvp = si_mat4x4_mul3(p, v, m);

        f32 speed = 0.005f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            lightPos.x -= speed;
        } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            lightPos.x += speed;
        } else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            lightPos.y += speed;
        } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            lightPos.y -= speed;
        } else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            lightPos.z += speed;
        } else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            lightPos.z -= speed;
        }

        glUniformMatrix4fv(mvpUni, 1, GL_FALSE, (f32*)mvp.v);
        glUniformMatrix4fv(modelUni, 1, GL_FALSE, (f32*)m.v);
        glUniform3fv(viewPosUni, 1, viewPos.v);
        glUniform3fv(lightPosUni, 1, lightPos.v);
        glUniform1f(uvScaleUni, 1.0f);

        glViewport(0, 0, winW, winH);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuse);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, ssbump);
        // glBindTexture(GL_TEXTURE_2D, normal);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, array_count(quad));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    si_free_primary_buffer(&mem.buffer);
}