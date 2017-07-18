// Author: Patrick Wieschollek <mail@patwie.com>
#ifndef META_H
#define META_H

#include <vector>
#include <opencv2/core.hpp>

template<typename T>
std::vector<float> linspace(T start_in, T end_in, const int num_in) {
    float start = static_cast<float>(start_in);
    float end = static_cast<float>(end_in);
    float num = static_cast<float>(num_in);
    float delta = (end - start) / (num - 1);

    std::vector<float> linspaced;
    for (int i = 0; i < num; ++i) {
        linspaced.push_back(start + delta * i);
    }
    // linspaced.push_back(end);
    return linspaced;
}


template<typename T>
T clip(T val, T lower, T upper) {
    val = (val < upper) ? val : upper - 1;
    val = (val < lower) ? lower : val;
    return val;
}


template<typename T>
T getMean(const std::vector<T>& images) {
    if (images.empty()) return T();

    cv::Mat accumulator(images[0].rows, images[0].cols, CV_32FC3, float(0.));
    cv::Mat temp;
    for (int i = 0; i < images.size(); ++i) {
        images[i].convertTo(temp, CV_32FC3);
        accumulator += temp;
    }

    accumulator.convertTo(accumulator, CV_8U, 1. / images.size());
    return accumulator;
}

template<typename T>
void pop_front(std::vector<T>& vec) {
    assert(!vec.empty());
    vec.erase(vec.begin());
}

cv::Mat scale(const cv::Mat input, float scale);

#endif