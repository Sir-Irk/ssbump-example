#include "types.h"
#include <stddef.h> 
#include "si_memory.h"
#include <stdio.h>

typedef struct read_file_result {
    ptrdiff_t contentsSize;
    void     *contents;
} read_file_result;

internal struct read_file_result
read_entire_file(const char *filepath, si_memory_arena *arena)
{
    struct read_file_result result = {};

    FILE *f = fopen(filepath, "rb");
    assert(f);

    ptrdiff_t size = 0;
    fseek(f, 0, SEEK_END);
    size = ftell(f) + 1;
    fseek(f, 0, SEEK_SET); // rewind
    void *contents = si_push_size(arena, size);
    fread(contents, 1, size - 1, f);
    fclose(f);
    ((char *)contents)[size - 1] = '\0';

    result.contentsSize = size;
    result.contents     = contents;

    return result;
}
