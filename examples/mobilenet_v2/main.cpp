#include "common/cpp/classification_demo.h"
#include "examples/mobilenet_v2/mobilenet_v2_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::mobilenetv2::Config());
}
