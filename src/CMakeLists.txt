project(IRAM)

# Generate json file for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native -funroll-loops -ftree-vectorize")

# Dependences
find_package(CBLAS REQUIRED)
find_package(LAPACKE REQUIRED)
find_package(OpenMP REQUIRED)
find_library(M_LIB m REQUIRED)

# Compiler
set(CMAKE_C_COMPILER "gcc")

# iram executable
add_executable(
    iram
    arnoldi.c    include/arnoldi.h
    file_test.c
    iram.c       include/iram.h
    kernel.c     include/kernel.h
    qr.c         include/qr.h
    ra.c         include/ra.h
    tools.c      include/tools.h
)
target_link_libraries(
    iram
    ${CBLAS_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${M_LIB}
    OpenMP::OpenMP_C
)
target_include_directories(
    iram PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)
