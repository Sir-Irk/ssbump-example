
WARNING_SUP="-Wno-unused-function -Wno-unused-variable -Wno-missing-braces"
LIBS="-lglfw -lGLU -lGL -lm"
FLAGS="-O0 -g -Wall -fno-math-errno -march=native"
clang ssbump.c $FLAGS -o ssbump.exe $LIBS $WARNING_SUP 
