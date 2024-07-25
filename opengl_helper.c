/*
Copyright (c) 2024 Jeremy Montgomery
Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include "types.h"
#include "glad/glad.h"
#include "stdio.h"
#include "si_memory.h"
#include "read_file.c"

void APIENTRY opengl_debug_callback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar *message,
    const void *userParam)
{
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH: {
            fprintf(stderr, "Severe Opengl Message: %s\n", message);
            assert(false);
        } break;
        case GL_DEBUG_SEVERITY_MEDIUM: {
            fprintf(stderr, "Medium severity Opengl Message: %s\n", message);
            assert(false);
        } break;
        case GL_DEBUG_SEVERITY_LOW: {
            fprintf(stderr, "Low severity Opengl Message: %s\n", message);
        } break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: {
        } break;
    }
}

internal const char *
get_gl_error_string(GLenum error)
{
    switch (error) {
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW: Given when a stack popping operation cannot be done because the stack is already at its lowest point";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY:";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST";
        default: {
            assert(!"Invalid opengl error code");
            return "Invalid opengl error code";
        } break;
    }
}

#define array_count(a) (sizeof(a) / sizeof(a[0]))

internal int
report_errors()
{
    const char *strings[5] = {};
    i32 maxStrings = array_count(strings);
    i32 numStrings = 0;
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR && numStrings < maxStrings) {
        strings[numStrings++] = get_gl_error_string(error);
    }

    for (i32 i = 0; i < numStrings; ++i) {
        fprintf(stderr, "%s\n", strings[i]);
    }

    return numStrings > 0;
}

internal GLint
get_uniform_location(GLuint program, const char* name){
    GLint result = glGetUniformLocation(program, name);
    if(result < 0){
        fprintf(stderr, "Failed to find uniform %s\n", name);
        //assert(0);
    }
    return result;
}

internal GLuint
create_shader(GLenum type, const char *code, si_memory_arena *arena)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &code, NULL);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        int infoSize;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoSize);

        si_temp_memory tempMem = si_start_temp_memory(arena);
        {
            char *log = si_push_array(arena, infoSize, char);
            glGetShaderInfoLog(shader, infoSize, NULL, log);
            fprintf(stderr, "%s\n%s\n", code, log);
        }
        si_pop_temp_memory(tempMem);
        assert(0);
    }

    assert(!report_errors());
    return shader;
}

internal GLuint
create_program(GLuint vShader, GLuint fShader, si_memory_arena *arena)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        int infoSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoSize);

        si_temp_memory tempMem = si_start_temp_memory(arena);
        {
            char *log = si_push_array(arena, infoSize, char);
            glGetProgramInfoLog(program, infoSize, NULL, log);
            fprintf(stderr, "%s\n", log);
        }
        si_pop_temp_memory(tempMem);
        assert(0);
    }

    assert(!report_errors());
    return program;
}

internal GLuint
create_program_from_files(const char *vertexShaderPath, const char *fragmentShaderPath, si_memory_arena *arena)
{
    struct read_file_result vCode = read_entire_file(vertexShaderPath, arena);
    struct read_file_result fCode = read_entire_file(fragmentShaderPath, arena);
    GLuint vShader = create_shader(GL_VERTEX_SHADER, (const char *)vCode.contents, arena);
    GLuint fShader = create_shader(GL_FRAGMENT_SHADER, (const char *)fCode.contents, arena);
    GLuint program = create_program(vShader, fShader, arena);
    glDeleteShader(vShader);
    glDeleteShader(fShader);
    assert(!report_errors());
    return program;
}

internal GLuint
create_program_from_strings(const char *vCode, const char *fCode, si_memory_arena *arena)
{
    GLuint vShader = create_shader(GL_VERTEX_SHADER, vCode, arena);
    GLuint fShader = create_shader(GL_FRAGMENT_SHADER, fCode, arena);
    GLuint program = create_program(vShader, fShader, arena);
    glDeleteShader(vShader);
    glDeleteShader(fShader);
    assert(!report_errors());
    return program;
}

internal GLuint
create_texture(const u32 *data, i32 w, i32 h, b32 useLinearColor)
{
    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    assert(!report_errors());

    if (useLinearColor) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
    assert(!report_errors());

    glGenerateMipmap(GL_TEXTURE_2D);
    assert(!report_errors());

    return result;
}
static void 
error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
	assert(0);
}
 
