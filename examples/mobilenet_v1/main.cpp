#include "common/cpp/classification_demo.h"
#include "examples/mobilenet_v1/mobilenet_v1_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::mobilenetv1::Config());
}
