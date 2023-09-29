#include <iostream>
#include <string>
#include <sstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

#include "sma.h"

float NET_OUTPUT_CONFIDENCE_THRESH = 0.3f;
float NMS_SUPPRESSION_THRESH = 0.3f;

float ASSUMED_OBJECT_WIDTH      = 150.f;   // [mm]
float VIDEO_CAMERA_FOCAL_LENGTH = 4.4f;    // [mm]
float VIDEO_CAMERA_PIXEL_SIZE   = 1.4e-3;  // [mm]

float triangleSimilarity(float focalLengthMm, float assumedWidthMm, float measuredWidthMm)
{
    return assumedWidthMm * focalLengthMm / measuredWidthMm;
}

int main()
{
    cv::VideoCapture cap("./doggo.mp4");
    cv::Mat frame, blob;

    auto net = cv::dnn::readNet("./mobilessd/deploy.prototxt", "./mobilessd/mobilenet_iter_73000.caffemodel");
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    cv::Size inputImgSize = cv::Size(300, 300);

    float *outputData;
    float top, left, bottom, right, width, height;
    int classId;
    float confidence;
    int boxIdx;

    SMA widthSMA = SMA(10);
    float depthM, widthMm, assumedWidth;

    for (;;)
    {
        if (!cap.read(frame))
            break;
        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        blob = cv::dnn::blobFromImage(frame, 1. / 255, inputImgSize);
        net.setInput(blob);
        net.forward(outs, outNames);

        // The shape of output blob is 1x1xNx7, where N is a number of detections and
        // 7 is a vector of each detection:
        //  [batchId, classId, confidence, left, top, right, bottom]
        for (size_t k = 0; k < outs.size(); k++)
        {
            outputData = (float *)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                confidence = outputData[i + 2];
                classId = (int)outputData[i + 1];
                if (confidence > NET_OUTPUT_CONFIDENCE_THRESH)
                {
                    left = outputData[i + 3] * frame.cols;
                    top = outputData[i + 4] * frame.rows;
                    right = outputData[i + 5] * frame.cols;
                    bottom = outputData[i + 6] * frame.rows;

                    width = right - left + 1;
                    height = bottom - top + 1;

                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        // Non-Max Supression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, NET_OUTPUT_CONFIDENCE_THRESH, NMS_SUPPRESSION_THRESH, indices);

        for (int i = 0; i < indices.size(); i++)
        {
            boxIdx = indices[i];
            cv::rectangle(frame, boxes[boxIdx], cv::Scalar(255, 255, 255), 1, 1, 0);

        }

        width = widthSMA.newVal(boxes[boxIdx].width);
        std::string label = "dog";

        if (widthSMA.m_filled)
        {
            widthMm = width * VIDEO_CAMERA_PIXEL_SIZE;
            depthM = 1e-3 * triangleSimilarity(VIDEO_CAMERA_FOCAL_LENGTH, ASSUMED_OBJECT_WIDTH, widthMm);
            std::ostringstream oss;
            oss.precision(2);
            oss << ": z=" << depthM << " [m]";
            label = label + oss.str();
        }

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);

        top = std::max(boxes[boxIdx].y, labelSize.height);
        cv::Point tl = cv::Point(boxes[boxIdx].x, top - labelSize.height);
        cv::Point br = cv::Point(boxes[boxIdx].x + labelSize.width, top);
        cv::rectangle(frame, tl, br, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(boxes[boxIdx].x, top), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);

        cv::imshow("doggo", frame);
        cv::waitKey(5);
    }

    std::cout << "done" << std::endl;
}