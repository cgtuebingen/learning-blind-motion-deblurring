// Author: Patrick Wieschollek <mail@patwie.com>
#include "meta.h"
#include "flow.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Flow::Flow() {
    int     numLevels = 3;
    double  pyrScale = 0.5;
    bool    fastPyramids = false;
    int     winSize = 15;
    int     numIters = 3;
    int     polyN = 5;
    double  polySigma = 1.2;
    int     flags = 0;
    flow_estimator = cv::cuda::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, winSize, numIters, polyN, polySigma, flags);

    initColorWheel();

}

void Flow::compute(cv::Mat a, cv::Mat b) {
    anchor = a;

    cv::Mat ag, bg;
    cv::cvtColor(a, ag, CV_BGR2GRAY);
    cv::cvtColor(b, bg, CV_BGR2GRAY);

    cv::cuda::GpuMat ad(ag);
    cv::cuda::GpuMat bd(bg);

    motion.create(a.size(), CV_32FC2);
    flow_estimator->calc(ad, bd, motion);

}

bool Flow::isFlowCorrect(cv::Point2f u) {
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}



void Flow::initColorWheel() {
    int k = 0;
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;

    for (int i = 0; i < RY; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);

    for (int i = 0; i < YG; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);

    for (int i = 0; i < GC; ++i, ++k)
        colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);

    for (int i = 0; i < CB; ++i, ++k)
        colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);

    for (int i = 0; i < BM; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);

    for (int i = 0; i < MR; ++i, ++k)
        colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);
}

cv::Vec3b Flow::computeColor(float fx, float fy) {

    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;


    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    cv::Vec3b pix;

    for (int b = 0; b < 3; b++) {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}


cv::Mat Flow::visualize() {
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(motion, planes);
    cv::Mat flowx(planes[0]);
    cv::Mat flowy(planes[1]);

    const cv::Mat_<float>& flowxx = flowx;
    const cv::Mat_<float>& flowyy = flowy;

    cv::Mat dst;
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    // determine motion range:
    double maxrad = 1;
    for (int y = 0; y < flowx.rows; ++y) {
        for (int x = 0; x < flowx.cols; ++x) {
            cv::Point2f u(flowxx(y, x), flowyy(y, x));
            if (!isFlowCorrect(u))
                continue;
            maxrad = std::max(maxrad, sqrt(u.x * u.x + u.y * u.y));
        }
    }


    for (int y = 0; y < flowx.rows; ++y) {
        for (int x = 0; x < flowx.cols; ++x) {
            cv::Point2f u(flowxx(y, x), flowyy(y, x));

            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
    return dst;
}



cv::Mat Flow::shift(const cv::Mat &img, float ratio) {
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(get(), planes);

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

            // make sure, we are sstill within the image
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

cv::Mat Flow::blur(const cv::Mat &img, std::vector<float> ratios) {
    cv::cuda::GpuMat planes[2];
    cv::cuda::split(get(), planes);

    // optical flow
    cv::Mat flowx(planes[0]);
    cv::Mat flowy(planes[1]);

    // resulting image is original + shifted values
    cv::Mat shifted_img, counter;
    img.convertTo(shifted_img, CV_32FC3);
    flowx.convertTo(counter, CV_32FC1);

    const int width = shifted_img.size().width;
    const int height = shifted_img.size().height;
    const int channels = shifted_img.channels();

    for (auto ratio : ratios) {
        // shifted versions of input image
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                // get offset for shifting in each direction
                int shift_x = flowx.at<float>(row, col);
                int shift_y = flowy.at<float>(row, col);

                // compute new coordinates
                int col_ = ratio * (col + shift_x) + (1.0 - ratio) * col;
                int row_ = ratio * (row + shift_y) + (1.0 - ratio) * row;

                // make sure, we are sstill within the image
                col_ = clip(col_, 0, width);
                row_ = clip(row_, 0, height);

                // add to previous image
                for (int z = 0; z < channels; ++z) {
                    // TODO: this might also overlap
                    shifted_img.at<cv::Vec3f>(row_, col_)[z] += float(img.at<cv::Vec3b>(row, col)[z]);

                }
            }
        }
    }
    return shifted_img;
}

const cv::cuda::GpuMat& Flow::get() const {
    return motion;
}