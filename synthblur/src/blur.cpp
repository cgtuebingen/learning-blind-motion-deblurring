// Author: Patrick Wieschollek <mail@patwie.com>
#include "blur.h"
#include "meta.h"

Blur::Blur() {}
cv::Mat Blur::shift(const cv::Mat &img, const Flow &d_flow, float ratio) {
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(d_flow.get(), planes);

    // optical flow
    cv::Mat flowx(planes[0]);
    cv::Mat flowy(planes[1]);

    // resulting image is original + shifted values
    cv::Mat shifted_img;
    img.convertTo(shifted_img, CV_32FC3);

    const int width = shifted_img.size().width;
    const int height = shifted_img.size().height;
    const int channels = shifted_img.channels();

    // shifted versions of input image
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // get offset for shifting in each direction
            int shift_x = flowx.at<float>(row, col);
            int shift_y = flowy.at<float>(row, col);

            // compute new coordinates
            int col_ = ratio * (col + shift_x) + (1.0 - ratio) * col;
            int row_ = ratio * (row + shift_y) + (1.0 - ratio) * row;

            // make sure, we are still within the image
            col_ = clip(col_, 0, width);
            row_ = clip(row_, 0, height);

            // add to previous image
            for (int z = 0; z < channels; ++z) {
                // TODO: this might also overlap
                shifted_img.at<cv::Vec3f>(row_, col_)[z] = float(img.at<cv::Vec3b>(row, col)[z]);

            }
        }
    }
    return shifted_img;
}