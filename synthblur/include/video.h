// Author: Patrick Wieschollek <mail@patwie.com>
#ifndef VIDEO_H
#define VIDEO_H

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>


class VideoReader {
  std::string path;
  cv::VideoCapture hnd;
 public:
  VideoReader(std::string fn);

  void jump(unsigned int idx);

  VideoReader& operator >> (cv::Mat& matrix);

  double fps() const;
  unsigned int frames() const;
  unsigned int frame() const;
  unsigned int  height() const;
  unsigned int  width() const;

};


class VideoWriter {
  std::string path;
  cv::VideoWriter hnd;
  int codec;
 public:
  VideoWriter(std::string fn, const int width, const int height, const float fps);
  VideoWriter& operator << (const cv::Mat& matrix);
};


#endif