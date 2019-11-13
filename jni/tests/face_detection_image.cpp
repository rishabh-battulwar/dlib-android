//============================================================================
// Name        : face_landmark_detection_ex.cpp
// Author      : Tzutalin
// Version     : 1.1
// Copyright   : Rishabh
// Description : face_detection_only_ex in C++, Google-Style

/* The face detector was created by
 using dlib's implementation.
 */
//============================================================================
#include <iostream>
#include <string>
#include <fstream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace dlib;
using namespace std;

int main(int argc, char** argv) {
    // Get the correct number of args
    if (argc != 3) {
        cout << "Call this program like this:" << endl;
        cout << "./face_landmark_detection_ex <image-path> <path-to-shape-predictor-data-file>" << endl;
        return -1;
    }

    // Create a file stream for the profile stats to be dumped out to 
    std::string directory("/data/local/tmp");
    char name[200];
    sprintf(name, "/detector_stats.txt");
    std::string out_file_path = directory + std::string(name);
    std::ofstream out(out_file_path, std::ios::out);
    char line[200];
        
    frontal_face_detector detector = get_frontal_face_detector();
    double detector_time;
    double landmark_time;
    
    auto printImageSize = [](const cv::Mat& img, const std::string& name="") {
        cout << name << " Image Size : " << img.size().width << ", " << img.size().height << "\n";
    };

    const std::string path = argv[1];
    // Read a image using opencv
    cv::Mat inputImage = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    // cv::Mat inputImage = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    // Check for invalid input
    if (!inputImage.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    printImageSize(inputImage, "Orig");
    
    // const int resizeWidth = 300;
    // const int resizeHeight = 168;
    // cv::resize(inputImage, inputImage, cv::Size(resizeWidth, resizeHeight));
    printImageSize(inputImage, "Resized");
    
    // // Convert mat to dlib's bgr pixel (shallow copy)
    dlib::cv_image<dlib::bgr_pixel> img(inputImage);
    dlib::array2d<unsigned char> imggray;
    dlib::assign_image(imggray, img);
    // dlib::cv_image<uchar> img(inputImage);

    cv::Rect cropROI;
    auto detectFaceFunc = [&]() {
        long t0 = cv::getTickCount();
        // Start detecting
        std::vector<rectangle> dets = detector(imggray);
        long t1 = cv::getTickCount();
        detector_time = (t1 - t0) / cv::getTickFrequency();
        cout << "take " << detector_time << " seconds, to find " << dets.size() << " faces" << "\n";
        cropROI = cv::Rect(
                    dets.at(0).tl_corner()(0), 
                    dets.at(0).tl_corner()(1), 
                    dets.at(0).width(), 
                    dets.at(0).height());
        return dets;
    };

    detectFaceFunc();
    
    cv::Mat croppedImage = inputImage(cropROI);
    printImageSize(croppedImage, "Cropped");
    img = dlib::cv_image<dlib::bgr_pixel>(croppedImage);
    dlib::assign_image(imggray, img);
    // img = dlib::cv_image<uchar>(croppedImage);

    auto dets = detectFaceFunc();

    shape_predictor sp;
    deserialize(argv[2]) >> sp;
    
    long t0 = cv::getTickCount();
    std::vector<full_object_detection> shapes;
    for (unsigned long j = 0; j < dets.size(); ++j) {
      full_object_detection shape = sp(img, dets[j]);
      // cout << "number of parts: " << shape.num_parts() << endl;
      // int x = shape.part(0).x();
      // int y = shape.part(0).y();
      // cout << "pixel position  0 index, x: " << x << endl;
      // cout << "pixel position  0 index, y: " << y << endl;
      shapes.push_back(shape);
    }
    long t1 = cv::getTickCount();
    landmark_time = (t1 - t0) / cv::getTickFrequency();
    cout << "takes " << landmark_time << " seconds, to find " << shapes.size() << " faces' keypoints" << "\n";


    sprintf(line, "[Detector Time] %2.6lf", detector_time);
    out << std::string(line) << std::endl;
    sprintf(line, "[Landmark Time] %2.6lf", landmark_time);
    out << std::string(line) << std::endl;
    
    out.close();

    return 0;
} 