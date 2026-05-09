#include "common/cpp/classification_demo.h"
#include "examples/resnet50/resnet50_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::resnet50::Config());
}
