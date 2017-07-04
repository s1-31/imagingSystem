#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <direct.h>
#include <sstream>
#include <fstream>
#include <chrono>

#include "isred.h"
#include "filter.h"

void convertImage(const cv::Mat, const cv::Mat, cv::Mat&, float);
void extract_notes(cv::Mat, cv::Mat&);
void add_notes(cv::Mat, cv::Mat, cv::Mat&);


void convertImage(const cv::Mat original, const cv::Mat withnotes, cv::Mat& converted, float reduction_rate) {
	cv::Mat small_original, small_withnotes;

	cv::resize(original, small_original, cv::Size(), reduction_rate, reduction_rate);
	cv::resize(withnotes, small_withnotes, cv::Size(), reduction_rate, reduction_rate);

	std::cout << "detector." << std::endl;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();

	auto time1 = std::chrono::system_clock::now();
	std::cout << "keypoint." << std::endl;
	std::vector<cv::KeyPoint> original_keypoint;
	cv::Mat original_descriptor;
	detector->detect(small_original, original_keypoint);
	detector->compute(small_original, original_keypoint, original_descriptor);

	std::vector<cv::KeyPoint> withnotes_keypoint;
	cv::Mat withnotes_descriptor;
	detector->detect(small_withnotes, withnotes_keypoint);
	detector->compute(small_withnotes, withnotes_keypoint, withnotes_descriptor);

	auto time2 = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() << "msec" << std::endl;

	std::cout << "match." << std::endl;
	std::vector<cv::DMatch> matches;
	cv::BFMatcher matcher;
	matcher.match(original_descriptor, withnotes_descriptor, matches);

	auto time3 = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2).count() << "msec" << std::endl;

	std::cout << "points." << std::endl;
	std::vector<cv::Vec2f> original_points(matches.size());
	std::vector<cv::Vec2f> withnotes_points(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
	{
		original_points[i][0] = original_keypoint[matches[i].queryIdx].pt.x / reduction_rate;
		original_points[i][1] = original_keypoint[matches[i].queryIdx].pt.y / reduction_rate;

		withnotes_points[i][0] = withnotes_keypoint[matches[i].trainIdx].pt.x / reduction_rate;
		withnotes_points[i][1] = withnotes_keypoint[matches[i].trainIdx].pt.y / reduction_rate;
	}
	auto time4 = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3).count() << "msec" << std::endl;

	std::cout << "homo.";
	cv::Mat homo = cv::findHomography(withnotes_points, original_points, CV_RANSAC);

	int width, height;
	width = static_cast<int>(original.cols);
	height = static_cast<int>(original.rows);

	std::cout << "convert." << std::endl;
	cv::Mat withnotes_converted;
	cv::warpPerspective(withnotes, withnotes_converted, homo, cv::Size(width, height));

	auto time5 = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4).count() << "msec" << std::endl;

	converted = withnotes_converted;
}

void extract_notes(cv::Mat image, cv::Mat& notes) {
	std::cout << "Start Extracting notes.";
	//double gm[] = { 0.0, 0.0, 0.0 };
	//double gstd[] = { 0.0, 0.0, 0.0 };
	//calc_global_edges_statistics(image, gm, gstd);
	//std::cout << "\n" << gm[0] << "," << gm[1] << "," << gm[2] << "," << std::endl;
	//std::cout << "\n" << gstd[0] << "," << gstd[1] << "," << gstd[2] << "," << std::endl;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			//if (isred(image.at<cv::Vec3b>(y, x), gm, gstd)) {
			if (isred(image.at<cv::Vec3b>(y, x))) {
				notes.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
			}
			else {
				notes.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}
		}
	}
}

void add_notes(cv::Mat original, cv::Mat notes, cv::Mat& withnotes) {
	for (int y = 0; y < withnotes.rows; y++) {
		for (int x = 0; x < withnotes.cols; x++) {
			if (isred(notes.at<cv::Vec3b>(y, x))) {
				withnotes.at<cv::Vec3b>(y, x) = notes.at<cv::Vec3b>(y, x);
			}
			else {
				withnotes.at<cv::Vec3b>(y, x) = original.at<cv::Vec3b>(y, x);
			}
		}
	}
}

