#include "common/cpp/classification_demo.h"
#include "examples/mobilenet_v3_small/mobilenet_v3_small_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::mobilenetv3small::Config());
}
