cmake_minimum_required(VERSION 3.0)
project(vSlamStereo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(EXE_NAME visualStereoSLAM.exe)
set(EXECUTABLE_OUTPUT_PATH "../")
set(SOURCES
	./src/point.cpp
	./src/frame.cpp
	./src/map.cpp
	./src/essentialMat.cpp
	./src/kdTree.cpp
	./src/display.cpp
	./src/slam.cpp
	./src/main.cpp
	./src/log.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")#"/usr/local/share/cmake-3.22/Modules")
include_directories(./include)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Torch REQUIRED)

find_package(Pangolin REQUIRED)
include_directories(/usr/local/include/pangolin)

find_package(G2O REQUIRED)
set(G2O_LIBS
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}

        ${G2O_SOLVER_CHOLMOD}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${G2O_SOLVER_DENSE}
        ${G2O_SOLVER_PCG}
        ${G2O_SOLVER_SLAM2D_LINEAR}
        ${G2O_SOLVER_EIGEN}

        ${G2O_TYPES_DATA}
        ${G2O_TYPES_ICP}
        ${G2O_TYPES_SBA}
        ${G2O_TYPES_SCLAM2D}
        ${G2O_TYPES_SIM3}
        ${G2O_TYPES_SLAM2D}
        ${G2O_TYPES_SLAM3D}
        )
include_directories(${G2O_INCLUDE_DIR})

add_executable(${EXE_NAME} ${SOURCES})

target_link_libraries(
	${EXE_NAME}
	${G2O_LIBS}
	${TORCH_LIBRARIES}
	${Pangolin_LIBRARIES}
	${OpenCV_LIBRARIES}
	)	
