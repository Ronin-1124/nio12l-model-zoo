#include "common/cpp/coco91_labels.h"

namespace mtkdemo {
namespace {

// Source: TensorFlow models research/object_detection/data/mscoco_complete_label_map.pbtxt
// display_name entries are ordered by id 0..90 (91 classes including background).
const char* kCoco91DisplayNames[] = {
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "12",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "26",
    "backpack",
    "umbrella",
    "29",
    "30",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "45",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "66",
    "dining table",
    "68",
    "69",
    "toilet",
    "71",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "83",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
};

}  // namespace

const std::vector<std::string>& Coco91Labels() {
  static const std::vector<std::string> kLabels = [] {
    std::vector<std::string> out;
    out.reserve(91);
    for (const char* name : kCoco91DisplayNames) {
      out.emplace_back(name);
    }
    return out;
  }();
  return kLabels;
}

}  // namespace mtkdemo
