// Author: Patrick Wieschollek <mail@patwie.com>

#include <iostream>
#include <vector>
#include "meta.h"
#include "flow.h"
#include "blur.h"
#include "video.h"


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <video>" << std::endl;
        return 0;
    }
    std::string video_path = argv[1];

    const int steps = 20;
    const float scaling_factor = 0.5;

    std::vector<float> ratios_before = linspace(1., 0., steps); ratios_before.pop_back();
    std::vector<float> ratios_after = linspace(0., 1., steps); pop_front(ratios_after);

    VideoReader video(video_path);
    std::cout << "simulate video with " << video.fps() * (1 + steps) << " fps" << std::endl;

    cv::Mat before_frame, sharp_frame, after_frame;
    video >> sharp_frame; sharp_frame = scale(sharp_frame, scaling_factor);
    video >> after_frame; after_frame = scale(after_frame, scaling_factor);

    Flow before_flow, after_flow;
    before_flow.compute(sharp_frame, after_frame);

    VideoWriter blurry_video(video_path + "_blurry.mp4", video.width() * scaling_factor, video.height() * scaling_factor, video.fps());
    VideoWriter sharp_video(video_path + "_sharp.mp4", video.width() * scaling_factor, video.height() * scaling_factor, video.fps());
    VideoWriter flow_video(video_path + "_flow.mp4", video.width() * scaling_factor, video.height() * scaling_factor, video.fps());

    // this is a re-write of the ugly RingBuffer implementation and does the job as well.
    for (int k = 0, k_e = video.frames() - 2; k < k_e; ++k) {

        // cyclic get frames
        before_frame = sharp_frame.clone();
        sharp_frame = after_frame.clone();
        video >> after_frame; after_frame = scale(after_frame, scaling_factor);

        // estimate FLOW
        before_flow.compute(sharp_frame, before_frame);
        after_flow.compute(sharp_frame, after_frame);

        // create blur
        std::vector<cv::Mat> subframes;
        int c = 0;
        for( auto ratio : ratios_before){
          subframes.push_back(before_flow.shift(sharp_frame, ratio));
        }
        for( auto ratio : ratios_after){
          subframes.push_back(after_flow.shift(sharp_frame, ratio));
        }
        cv::Mat blurry_frame = getMean(subframes);

        // write to video
        blurry_video << blurry_frame;
        sharp_video << sharp_frame;
        flow_video << before_flow.visualize();

        std::cout << k << " / "<< k_e << std::endl;
          
   
    }

    return 0;
}
