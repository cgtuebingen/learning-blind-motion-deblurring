// Author: Patrick Wieschollek <mail@patwie.com>
#include <iostream>
#include "video.h"

VideoReader::VideoReader(std::string fn) : hnd(fn), path(fn) {
  std::cout << "read video " << fn << std::endl
            << "  fps:    " << fps() << std::endl
            << "  frames: " << frames() << std::endl
            << "  shapes: H:" << height() << " W:" << width() << std::endl;

}

void VideoReader::jump(unsigned int idx) {
  hnd.set(CV_CAP_PROP_POS_FRAMES, idx);
  std::cout << "jump to frame " << frame() << std::endl;

}

double VideoReader::fps() const { return hnd.get(CV_CAP_PROP_FPS);}
unsigned int  VideoReader::width() const { return hnd.get(CV_CAP_PROP_FRAME_WIDTH);}
unsigned int  VideoReader::height() const { return hnd.get(CV_CAP_PROP_FRAME_HEIGHT);}
unsigned int VideoReader::frames() const { return hnd.get(CV_CAP_PROP_FRAME_COUNT);}
unsigned int VideoReader::frame() const { return hnd.get(CV_CAP_PROP_POS_FRAMES);}


VideoReader& VideoReader::operator >> (cv::Mat& matrix) {
  hnd >> matrix;
  return *this;
}


VideoWriter::VideoWriter(std::string fn, const int width, const int height, const float fps) {
  codec = CV_FOURCC('m', 'p', '4', 'v');
  cv::Size S = cv::Size(width, height);
  std::cout << "write video " << fn << std::endl
            << "  fps:    " << fps << std::endl
            << "  shapes: H:" << height << " W:" << width << std::endl;
  hnd.open(fn, codec, fps, S, true);
}

VideoWriter& VideoWriter::operator << (const cv::Mat& matrix) {
  cv::Mat frame;
  matrix.convertTo(frame, CV_8UC3);
  hnd << frame;
  return *this;
}