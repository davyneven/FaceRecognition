cmake_minimum_required(VERSION 2.8)

project(RecognitionProject)

include(/usr/dlib-18.10/dlib/cmake)



find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( faceRecognition main.cpp FaceDetector.cpp Recognition.cpp)

target_link_libraries(faceRecognition dlib ${OpenCV_LIBS} )
set_target_properties(faceRecognition PROPERTIES COMPILE_FLAGS "-D_LINUX -lX11 -D_REENTRANT -Wall  -O3 -march=native -msse2avx")

#add_executable(Recognizer mainRecognizer.cpp FaceDetector.cpp Recognition.cpp)
#target_link_libraries(Recognizer dlib ${OpenCV_LIBS} )
#set_target_properties(Recognizer PROPERTIES COMPILE_FLAGS "-D_LINUX -lX11 -D_REENTRANT -Wall  -O3 -march=native -msse2avx")
