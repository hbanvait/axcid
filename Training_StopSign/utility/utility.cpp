/*
 * Copyright (c) 2012-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <unistd.h>

#include <cstdio>

#include <iostream>
#include <deque>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

////////////////////////////////////
// ImageSource

namespace
{
    class ImageSource : public FrameSource
    {
    public:
        explicit ImageSource(const string& fileName, int flags = 1);

        void next(Mat& frame);

        void reset();

    private:
        Mat img_;
    };

    ImageSource::ImageSource(const string& fileName, int flags)
    {
        img_ = imread(fileName, flags);

        if (img_.empty())
            THROW_EXCEPTION("Can't open " << fileName << " image");
    }

    void ImageSource::next(Mat& frame)
    {
        frame = img_;
    }

    void ImageSource::reset()
    {
    }
}

Ptr<FrameSource> FrameSource::image(const string& fileName, int flags)
{
    return new ImageSource(fileName, flags);
}

////////////////////////////////////
// VideoSource

namespace
{
    class VideoSource : public FrameSource
    {
    public:
        explicit VideoSource(const string& fileName);

        void next(Mat& frame);

        void reset();

    protected:
        string fileName_;
        VideoCapture vc_;
    };

    VideoSource::VideoSource(const string& fileName) : fileName_(fileName)
    {
        if (!vc_.open(fileName))
            THROW_EXCEPTION("Can't open " << fileName << " video");
    }

    void VideoSource::next(Mat& frame)
    {
        vc_ >> frame;

        if (frame.empty())
        {
            reset();
            vc_ >> frame;
        }
    }

    void VideoSource::reset()
    {
        vc_.release();
        vc_.open(fileName_);
    }
}

Ptr<FrameSource> FrameSource::video(const string& fileName)
{
    return new VideoSource(fileName);
}

////////////////////////////////////
// CameraSource

namespace
{
    class CameraSource : public FrameSource
    {
    public:
        explicit CameraSource(int device, int width = -1, int height = -1);

        void next(Mat& frame);

        void reset();

    private:
        VideoCapture vc_;
    };

    CameraSource::CameraSource(int device, int width, int height)
    {
        if (!vc_.open(device))
            THROW_EXCEPTION("Can't open camera with ID = " << device);

        if (width > 0)
            vc_.set(CV_CAP_PROP_FRAME_WIDTH, width);

        if (height > 0)
            vc_.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    }

    void CameraSource::next(Mat& frame)
    {
        vc_ >> frame;
    }

    void CameraSource::reset()
    {
    }
}

Ptr<FrameSource> FrameSource::camera(int device, int width, int height)
{
    return new CameraSource(device, width, height);
}

////////////////////////////////////
// PairFrameSource

namespace
{
    class PairFrameSource_2 : public PairFrameSource
    {
    public:
        PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1);

        void next(Mat& frame0, Mat& frame1);

        void reset();

    private:
        Ptr<FrameSource> source0_;
        Ptr<FrameSource> source1_;
    };

    PairFrameSource_2::PairFrameSource_2(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1) :
        source0_(source0), source1_(source1)
    {
        CV_Assert( !source0_.empty() );
        CV_Assert( !source1_.empty() );
    }

    void PairFrameSource_2::next(Mat& frame0, Mat& frame1)
    {
        source0_->next(frame0);
        source1_->next(frame1);
    }

    void PairFrameSource_2::reset()
    {
        source0_->reset();
        source1_->reset();
    }

    class PairFrameSource_1 : public PairFrameSource
    {
    public:
        PairFrameSource_1(const Ptr<FrameSource>& source, int offset);

        void next(Mat& frame0, Mat& frame1);

        void reset();

    private:
        Ptr<FrameSource> source_;
        int offset_;
        deque<Mat> frames_;
    };

    PairFrameSource_1::PairFrameSource_1(const Ptr<FrameSource>& source, int offset) : source_(source), offset_(offset)
    {
        CV_Assert( !source_.empty() );

        reset();
    }

    void PairFrameSource_1::next(Mat& frame0, Mat& frame1)
    {
        source_->next(frame1);
        frames_.push_back(frame1.clone());
        frame0 = frames_.front();
        frames_.pop_front();
    }

    void PairFrameSource_1::reset()
    {
        source_->reset();

        frames_.clear();
        Mat temp;
        for (int i = 0; i < offset_; ++i)
        {
            source_->next(temp);
            frames_.push_back(temp.clone());
        }
    }
}

Ptr<PairFrameSource> PairFrameSource::create(const Ptr<FrameSource>& source0, const Ptr<FrameSource>& source1)
{
    return new PairFrameSource_2(source0, source1);
}

Ptr<PairFrameSource> PairFrameSource::create(const Ptr<FrameSource>& source, int offset)
{
    return new PairFrameSource_1(source, offset);
}

////////////////////////////////////
// Auxiliary functions

void checkCudaDevice(int deviceID)
{
    const int num_devices = getCudaEnabledDeviceCount();
    if (num_devices <= 0)
        THROW_EXCEPTION("No GPU found or the OpenCV library was compiled without CUDA support");

    if (deviceID < 0 || deviceID >= num_devices)
        THROW_EXCEPTION("Incorrect device ID : " << deviceID);

    DeviceInfo dev_info(deviceID);
    if (!dev_info.isCompatible())
        THROW_EXCEPTION("GPU module wasn't built for GPU #" << deviceID << " " << dev_info.name() << ", CC " << dev_info.majorVersion() << '.' << dev_info.minorVersion());

    cout << "Initializing device... \n" << endl;
    setDevice(deviceID);
    printShortCudaDeviceInfo(deviceID);

    cout << endl;
}

static std::string getExecPath()
{
    char buf[1024];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);

    if (len != -1)
    {
        buf[len] = '\0';

        std::string full_exec_name = buf;

        size_t delimiter_pos = full_exec_name.find_last_of('/');

        if (delimiter_pos != std::string::npos)
            return full_exec_name.substr(0, delimiter_pos + 1);
    }

    return "./";
}

string findDemoFilePath(const string& filename)
{
    std::string exec_path = getExecPath();

    string searchPath[] =
    {
        exec_path + std::string("data/"),
        exec_path + std::string("../data/"),
        exec_path + std::string("../../data/"),
        exec_path + std::string("../../../data/"),
        exec_path + std::string("../../../../data/"),
        exec_path + std::string("../../../../../data/"),
        std::string("/usr/share/visionworks/demos/data/")
    };

    // Loop over all search paths and return the first hit
    for (size_t i = 0; i < sizeof(searchPath) / sizeof(searchPath[0]); ++i)
    {
        std::string path = searchPath[i];

        // Test if the file exists
        path.append(filename);

        FILE *fp = fopen(path.c_str(), "rb");

        if (fp != NULL)
        {
            fclose(fp);
            return path;
        }

        if (fp)
        {
            fclose(fp);
        }
    }

    THROW_EXCEPTION("Can't find " << filename);

    // File not found
    return string();
}

void printText(Mat& img, const string& msg, int lineOffsY, Scalar fontColor, double fontScale)
{
    const int fontFace = FONT_HERSHEY_DUPLEX;
    const int fontThickness = 2;

    const Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;

    putText(img, msg, org, fontFace, fontScale, Scalar(0,0,0,255), 5 * fontThickness / 2, 16);
    putText(img, msg, org, fontFace, fontScale, fontColor, fontThickness, 16);
}
