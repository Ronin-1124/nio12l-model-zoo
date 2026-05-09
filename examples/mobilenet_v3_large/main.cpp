#include "common/cpp/classification_demo.h"
#include "examples/mobilenet_v3_large/mobilenet_v3_large_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::mobilenetv3large::Config());
}
