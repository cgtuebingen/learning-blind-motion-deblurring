// Author: Patrick Wieschollek <mail@patwie.com>
#include "meta.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat scale(const cv::Mat input, const float scale) {
  cv::Mat small;
  cv::resize(input, small, cv::Size(), scale, scale, cv::INTER_CUBIC);
  return small;
}