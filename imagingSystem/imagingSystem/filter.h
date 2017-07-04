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

void calc_global_statistics(cv::Mat, double*, double*);
int channel_based_sigmoid(int);
void unsharp_masking(cv::Mat, cv::Mat&, float);
void removeShadow(cv::Mat&);

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


int channel_based_sigmoid(int v) {
	return static_cast<int>(255.0 / (1.0 + exp(-1.0*(v - 150))));
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

void removeShadow(cv::Mat& img) {
	cv::Mat gray, adaptive;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::adaptiveThreshold(gray, adaptive, 255,
		cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::ThresholdTypes::THRESH_BINARY,
		9, 12);
	//cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
	//cv::morphologyEx(adaptive, adaptive, cv::MorphTypes::MORPH_CLOSE, kernel);
	//cv::morphologyEx(adaptive, adaptive, cv::MorphTypes::MORPH_OPEN, kernel);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (adaptive.at<uchar>(y, x) == 255) {
				img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}
		}
	}
}