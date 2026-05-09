cmake_minimum_required(VERSION 3.16)

project(mtk_genio_1200_model_zoo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
  mtk_demo_common
  common/cpp/coco_labels.cpp
  common/cpp/classification_demo.cpp
  common/cpp/coco91_labels.cpp
  common/cpp/dota_labels.cpp
  common/cpp/detection_utils.cpp
  common/cpp/imagenet_labels.cpp
  common/cpp/image_io.cpp
  common/cpp/neuron_runtime.cpp
  common/cpp/preprocess.cpp
  common/cpp/ssd_mobilenet_v2_postprocess.cpp
  common/cpp/stb_impl.cpp
  common/cpp/yolo11_postprocess.cpp
  common/cpp/yolo12_postprocess.cpp
  common/cpp/yolo26_postprocess.cpp
  common/cpp/yolo26n-pose_postprocess.cpp
  common/cpp/yolo26n-seg_postprocess.cpp
  common/cpp/yolo26n-obb_postprocess.cpp
  common/cpp/yolov8n-seg_postprocess.cpp
  common/cpp/yolov8n-obb_postprocess.cpp
  common/cpp/yolov8n-pose_postprocess.cpp
  common/cpp/yolov10_postprocess.cpp
  common/cpp/yolov5_postprocess.cpp
  common/cpp/yolov8_postprocess.cpp
)

target_include_directories(
  mtk_demo_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/neuron/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

target_link_libraries(mtk_demo_common PUBLIC neuronusdk_runtime.mtk)

add_executable(yolov5s_demo examples/yolov5s/main.cpp)
target_link_libraries(yolov5s_demo PRIVATE mtk_demo_common)

add_executable(yolov8n_demo examples/yolov8n/main.cpp)
target_link_libraries(yolov8n_demo PRIVATE mtk_demo_common)

add_executable(yolo11n_demo examples/yolo11n/main.cpp)
target_link_libraries(yolo11n_demo PRIVATE mtk_demo_common)

add_executable(yolov10n_demo examples/yolov10n/main.cpp)
target_link_libraries(yolov10n_demo PRIVATE mtk_demo_common)

add_executable(yolo26n_demo examples/yolo26n/main.cpp)
target_link_libraries(yolo26n_demo PRIVATE mtk_demo_common)

add_executable(mobilenet_v1_demo examples/mobilenet_v1/main.cpp)
target_link_libraries(mobilenet_v1_demo PRIVATE mtk_demo_common)

add_executable(mobilenet_v2_demo examples/mobilenet_v2/main.cpp)
target_link_libraries(mobilenet_v2_demo PRIVATE mtk_demo_common)

add_executable(ssd_mobilenet_v2_demo examples/ssd_mobilenet_v2/main.cpp)
target_link_libraries(ssd_mobilenet_v2_demo PRIVATE mtk_demo_common)

add_executable(efficientnet_b0_classification_demo
  examples/efficientnet_b0_classification/main.cpp)
target_link_libraries(efficientnet_b0_classification_demo PRIVATE mtk_demo_common)

add_executable(resnet50_demo examples/resnet50/main.cpp)
target_link_libraries(resnet50_demo PRIVATE mtk_demo_common)

add_executable(mobilenet_v3_large_demo examples/mobilenet_v3_large/main.cpp)
target_link_libraries(mobilenet_v3_large_demo PRIVATE mtk_demo_common)

add_executable(mobilenet_v3_small_demo examples/mobilenet_v3_small/main.cpp)
target_link_libraries(mobilenet_v3_small_demo PRIVATE mtk_demo_common)

add_executable(yolo12n_demo examples/yolo12n/main.cpp)
target_link_libraries(yolo12n_demo PRIVATE mtk_demo_common)

add_executable(yolov8n-seg_demo examples/yolov8n-seg/main.cpp)
target_link_libraries(yolov8n-seg_demo PRIVATE mtk_demo_common)

add_executable(yolo11n-seg_demo examples/yolo11n-seg/main.cpp)
target_link_libraries(yolo11n-seg_demo PRIVATE mtk_demo_common)

add_executable(yolo26n-seg_demo examples/yolo26n-seg/main.cpp)
target_link_libraries(yolo26n-seg_demo PRIVATE mtk_demo_common)

add_executable(yolov8n-pose_demo examples/yolov8n-pose/main.cpp)
target_link_libraries(yolov8n-pose_demo PRIVATE mtk_demo_common)

add_executable(yolov8n-obb_demo examples/yolov8n-obb/main.cpp)
target_link_libraries(yolov8n-obb_demo PRIVATE mtk_demo_common)

add_executable(yolo11n-obb_demo examples/yolo11n-obb/main.cpp)
target_link_libraries(yolo11n-obb_demo PRIVATE mtk_demo_common)

add_executable(yolo26n-obb_demo examples/yolo26n-obb/main.cpp)
target_link_libraries(yolo26n-obb_demo PRIVATE mtk_demo_common)

add_executable(yolo11n-pose_demo examples/yolo11n-pose/main.cpp)
target_link_libraries(yolo11n-pose_demo PRIVATE mtk_demo_common)

add_executable(yolo26n-pose_demo examples/yolo26n-pose/main.cpp)
target_link_libraries(yolo26n-pose_demo PRIVATE mtk_demo_common)
