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

bool isred_based_pstd(int r, int b, int g) {
	double threshold = 0.02;
	double vlth = 0.1;

	double db = b / 255.0;
	double dg = g / 255.0;
	double dr = r / 255.0;

	double m = (dr + dg + db) / 3.0;
	double std = ((db - m)*(db - m) + (dg - m)*(dg - m) + (dr - m)*(dr - m)) / m;
	if (std > threshold && dr > dg && dr > db && sqrt(std*m) > vlth) {
		return true;
	}
	return false;
}

void calc_global_statistics(cv::Mat image, double* m, double* std) {
	for (int i = 0; i < 3; i++) {
		m[i] = 0.0;
		std[i] = 0.0;
	}
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			cv::Vec3b p = image.at<cv::Vec3b>(y, x);
			for (int i = 0; i < 3; i++) {
				m[i] += static_cast<double>(p(i));
				std[i] += static_cast<double>(p(i))*static_cast<double>(p(i));
			}
		}
	}
	for (int i = 0; i < 3; i++) {
		m[i] = m[i] / (image.rows*image.cols);
		std[i] = sqrt(std[i] / (image.rows*image.cols) - m[i] * m[i]);
	}
}

bool isred_based_global_std(int* p, double* gm, double* gstd) {
	double threshold = 0.3*0.3*0.3;
	double score = 1.0;
	for (int i = 0; i < 3; i++) {
		score *= (p[i] - gm[i]) / gstd[i];
	}
	if (score > threshold) {
		return true;
	}
	return false;
}

/*
bool isred(cv::Vec3b pixel) {
int b = pixel(0);
int g= pixel(1);
int r= pixel(2);

return isred_based_pstd(r, g, b);
}
*/

bool isred(cv::Vec3b pixel, double* gm, double* gstd) {
	int b = pixel(0);
	int g = pixel(1);
	int r = pixel(2);
	int p[] = { b, g, r };

	return isred_based_global_std(p, gm, gstd);
}

void unsharp_masking(cv::Mat src_mat, cv::Mat& dst_mat, float k) {
	IplImage src = src_mat;
	IplImage dst = dst_mat;
	//カーネルの設定
	float KernelData[] = {
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
		-k / 9.0f, 1 + (8 * k) / 9.0f,  -k / 9.0f,
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
	};
	//カーネルの配列をCvMatへ変換
	CvMat kernel = cvMat(3, 3, CV_32F, KernelData);
	//フィルタ処理
	cvFilter2D(&src, &dst, &kernel);
}

int channel_based_sigmoid(int v) {
	return static_cast<int>(255.0 / (1.0 + exp(-1.0*(v - 150))));
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
			//if (isred(image.at<cv::Vec3b>(y, x))) {
			if (isred(image.at<cv::Vec3b>(y, x), gm, gstd)) {
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
			if (notes.at<cv::Vec3b>(y, x)(0) < 255) {
				withnotes.at<cv::Vec3b>(y, x) = notes.at<cv::Vec3b>(y, x);
			}
			else {
				withnotes.at<cv::Vec3b>(y, x) = original.at<cv::Vec3b>(y, x);
			}
		}
	}
}

