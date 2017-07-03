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

#include "isred.h"
#include "filter.h"

void convertImage(const cv::Mat, const cv::Mat, cv::Mat&);
void extract_notes(cv::Mat, cv::Mat&);
void add_notes(cv::Mat, cv::Mat, cv::Mat&);

void convertImage(const cv::Mat original, const cv::Mat withnotes, cv::Mat& converted) {
	std::cout << "detector.";
	//cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();

	std::cout << "keypoint.";
	std::vector<cv::KeyPoint> original_keypoint;
	cv::Mat original_descriptor;
	detector->detect(original, original_keypoint);
	detector->compute(original, original_keypoint, original_descriptor);

	std::vector<cv::KeyPoint> withnotes_keypoint;
	cv::Mat withnotes_descriptor;
	detector->detect(withnotes, withnotes_keypoint);
	detector->compute(withnotes, withnotes_keypoint, withnotes_descriptor);

	std::cout << "match.";
	std::vector<cv::DMatch> matches;

	std::cout << "matcher.";
	cv::BFMatcher matcher;
	matcher.match(original_descriptor, withnotes_descriptor, matches);

	std::cout << "points.";
	std::vector<cv::Vec2f> original_points(matches.size());
	std::vector<cv::Vec2f> withnotes_points(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
	{

		original_points[i][0] = original_keypoint[matches[i].queryIdx].pt.x;
		original_points[i][1] = original_keypoint[matches[i].queryIdx].pt.y;

		withnotes_points[i][0] = withnotes_keypoint[matches[i].trainIdx].pt.x;
		withnotes_points[i][1] = withnotes_keypoint[matches[i].trainIdx].pt.y;
	}

	//Mat matchedImg;
	//drawMatches(src[0], keypoints[0], src[1], keypoints[1], matches, matchedImg);
	//imshow("draw img", matchedImg);
	//waitKey(0);

	std::cout << "homo.";
	cv::Mat homo = cv::findHomography(withnotes_points, original_points, CV_RANSAC);

	int width, height;
	width = static_cast<int>(original.cols);
	height = static_cast<int>(original.rows);

	std::cout << "convert.";
	cv::Mat withnotes_converted;
	cv::warpPerspective(withnotes, withnotes_converted, homo, cv::Size(width, height));

	converted = withnotes_converted;
}

void extract_notes(cv::Mat image, cv::Mat& notes) {
	std::cout << "Start Extracting notes.";
	//cv::Mat sharp_image = image.clone();
	//unsharp_masking(image, sharp_image, 3);
	//image = sharp_image;
	double gm[] = { 0.0, 0.0, 0.0 };
	double gstd[] = { 0.0, 0.0, 0.0 };
	calc_global_statistics(image, gm, gstd);
	std::cout << "\n", gm[0], gm[1], gm[2], "\n";
	std::cout << "\n", gstd[0], gstd[1], gstd[2], "\n";

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (isred(image.at<cv::Vec3b>(y, x))) {
			//if (isred(image.at<cv::Vec3b>(y, x), gm, gstd)) {
				//TODO: sigmoidかけるとか？
				//赤いピクセルの統計情報が取れれば、もっとやりようはある。
				notes.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
				//notes.at<cv::Vec3b>(y, x) = cv::Vec3b(channel_based_sigmoid(b), channel_based_sigmoid(g), channel_based_sigmoid(r));
				//notes.at<cv::Vec3b>(y, x) = cv::Vec3b(50, 50, 200);
				/*
				cv::Vec3b p = image.at<cv::Vec3b>(y, x);
				int b = p(0);
				int g = p(1);
				int r = p(2);
				cv::Vec3b filtered_p = cv::Vec3b(channel_based_sigmoid(b), channel_based_sigmoid(g), channel_based_sigmoid(r));
				if (isred(filtered_p)) {
				notes.at<cv::Vec3b>(y, x) = filtered_p;
				}
				else {
				notes.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
				}
				*/
			}
			else {
				notes.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
				// notes.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}
		}
	}
}

void add_notes(cv::Mat original, cv::Mat notes, cv::Mat& withnotes) {
	for (int y = 0; y < withnotes.rows; y++) {
		for (int x = 0; x < withnotes.cols; x++) {
			// cv::add(original.at<cv::Vec3b>(y, x), notes.at<cv::Vec3b>(y, x), withnotes.at<cv::Vec3b>(y, x));
			if (isred(notes.at<cv::Vec3b>(y, x))) {
				withnotes.at<cv::Vec3b>(y, x) = notes.at<cv::Vec3b>(y, x);
			}
			else {
				withnotes.at<cv::Vec3b>(y, x) = original.at<cv::Vec3b>(y, x);
			}
		}
	}
}

