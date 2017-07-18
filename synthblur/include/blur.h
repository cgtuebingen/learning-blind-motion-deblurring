// Author: Patrick Wieschollek <mail@patwie.com>
#ifndef BLUR_H
#define BLUR_H

#include <opencv2/opencv.hpp>

#include "flow.h"

class Blur
{
public:
  Blur();
  cv::Mat shift(const cv::Mat &anchor_, const Flow &flow_, float ratio);
};

#endif