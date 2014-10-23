/*
 * Copyright (c) 2012-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

#include <string>
#include <sstream>
#include <stdexcept>

#include <opencv2/core/core.hpp>

////////////////////////////////////
// FrameSource

class FrameSource
{
public:
    virtual ~FrameSource() {}

    virtual void next(cv::Mat& frame) = 0;

    virtual void reset() = 0;

    static cv::Ptr<FrameSource> image(const std::string& fileName, int flags = 1);
    static cv::Ptr<FrameSource> video(const std::string& fileName);
    static cv::Ptr<FrameSource> camera(int device, int width = -1, int height = -1);
};

class PairFrameSource
{
public:
    virtual ~PairFrameSource() {}

    virtual void next(cv::Mat& frame0, cv::Mat& frame1) = 0;

    virtual void reset() = 0;

    static cv::Ptr<PairFrameSource> create(const cv::Ptr<FrameSource>& source0, const cv::Ptr<FrameSource>& source1);
    static cv::Ptr<PairFrameSource> create(const cv::Ptr<FrameSource>& source, int offset);
};

////////////////////////////////////
// Auxiliary functions

void checkCudaDevice(int deviceID);

std::string findDemoFilePath(const std::string& filename);

void printText(cv::Mat& img, const std::string& msg, int lineOffsY, cv::Scalar fontColor = CV_RGB(118, 185, 0), double fontScale = 0.8);

#define THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg ; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

#endif // __UTILITY_HPP__
