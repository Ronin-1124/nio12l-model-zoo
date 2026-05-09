#include "common/cpp/classification_demo.h"
#include "examples/efficientnet_b0_classification/efficientnet_b0_classification_config.h"

int main(int argc, char** argv) {
  return mtkdemo::RunClassificationDemo(argc, argv, mtkdemo::efficientnetb0::Config());
}
