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

double mean_squeare_error(cv::Mat origin, cv::Mat edited, int iterate_num, std::string out_dir) {
	cv::Mat gray_o, gray_e, gray16s_o, gray16s_e;
	cv::cvtColor(origin, gray_o, CV_BGR2GRAY);
	cv::cvtColor(edited, gray_e, CV_BGR2GRAY);
	gray_o.convertTo(gray16s_o, CV_16S);
	gray_e.convertTo(gray16s_e, CV_16S);

	gray16s_o -= gray16s_e;

	cv::Scalar s = cv::sum(gray16s_o.mul(gray16s_o));

	cv::Mat gray8u_o;
	gray16s_o.convertTo(gray8u_o, CV_8U);
	imshow("result img4", gray8u_o);
	cv::imwrite(out_dir + "\\diff" + std::to_string(iterate_num+1) + ".jpg", gray8u_o);
	//cv::waitKey(0);

	edited = gray16s_o;

	return s[0] / gray16s_o.rows / gray16s_o.cols;
}

std::vector<std::string> split(const std::string &str, char sep)
{
	std::vector<std::string> v;
	auto first = str.begin(); 
	while (first != str.end()) {        
		auto last = first;               
		while (last != str.end() && *last != sep)       
			++last;
		v.push_back(std::string(first, last));   
		if (last != str.end())
			++last;
		first = last;
	}
	return v;
}

void main() {
	using cv::imread;
	using cv::Mat;
	using std::vector;
	using cv::imshow;
	using cv::waitKey;
	using cv::Size;
	using cv::Vec3b;

	Mat src[2];
	Mat gray[2];
	Mat result;

	std::string origin_name = "653-origin.jpg";
	std::string photo_name = "653-top-notes.jpg";

	//std::string origin_name = "lena.jpg";
	//std::string photo_name = "printed_lena.jpg";

	// 結果保存用のフォルダとファイル
	char out_dir[] = "";
	std::string out_file = "output.txt";
	std::ofstream writing_file;

	time_t now = time(NULL);
	struct tm *pnow = localtime(&now);
	sprintf(out_dir, "results\\%s%02d%02d%02d%02d", 
		split(photo_name, '.')[0].c_str(), pnow->tm_mon + 1, pnow->tm_mday, pnow->tm_hour, pnow->tm_min);
	_mkdir(out_dir);
	writing_file.open(std::string(out_dir)+"\\"+out_file, std::ios::out);

	src[1] = imread(origin_name);
	src[0] = imread(photo_name);

	cv::resize(src[0], src[0], cv::Size(), 0.5, 0.5);
	cv::resize(src[1], src[1], cv::Size(), 0.5, 0.5);
	
	//src[0] = imread("lena.jpg");
	//src[1] = imread("printed_lena.jpg");

	int iter_num = 5;

	for (int j = 0; j < iter_num; j++) {
		cv::cvtColor(src[0], gray[0], CV_BGR2GRAY);
		cv::cvtColor(src[1], gray[1], CV_BGR2GRAY);

		// gray 画像にする
		//gray[0].copyTo(src[0]);

		//cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
		cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();

		vector<cv::KeyPoint> keypoints[2];
		Mat descriptors[2];
		for (int i = 0; i < 2; i++) {
			detector->detect(gray[i], keypoints[i]);
			detector->compute(gray[i], keypoints[i], descriptors[i]);
		}

		vector<cv::DMatch> matches;

		cv::BFMatcher matcher;
		matcher.match(descriptors[0], descriptors[1], matches);

		vector<cv::Vec2f> points1(matches.size());
		vector<cv::Vec2f> points2(matches.size());

		for (size_t i = 0; i < matches.size(); ++i)
		{
			points1[i][0] = keypoints[0][matches[i].queryIdx].pt.x;
			points1[i][1] = keypoints[0][matches[i].queryIdx].pt.y;

			points2[i][0] = keypoints[1][matches[i].trainIdx].pt.x;
			points2[i][1] = keypoints[1][matches[i].trainIdx].pt.y;
		}

		Mat matchedImg;
		drawMatches(src[0], keypoints[0], src[1], keypoints[1], matches, matchedImg);
		imshow("draw img", matchedImg);
		//waitKey(0);

		Mat homo = cv::findHomography(points1, points2, CV_RANSAC);
		int width, height;
		if (src[1].cols < src[0].cols) {
			width = static_cast<int>(src[1].cols);
			height = static_cast<int>(src[1].rows);
		}
		else {
			width = static_cast<int>(src[0].cols);
			height = static_cast<int>(src[0].rows);
		}
		cv::warpPerspective(src[0], result, homo, Size(width, height));

		imshow("result img", result);
		cv::imwrite(std::string(out_dir) + "\\result" + std::to_string(j+1) + ".jpg", result);
		//waitKey(0);

		/*for (int y = 0; y < src[1].rows; y++) {
		for (int x = 0; x < src[1].cols; x++) {
		cv::absdiff(result.at<Vec3b>(y, x), src[1].at<Vec3b>(y, x), result.at<Vec3b>(y, x));
		}
		}

		cv::Mat crop = result(cv::Rect(0, 0, src[1].cols, src[1].rows));

		imshow("substracttion", crop);
		waitKey(0);*/

		double mse = mean_squeare_error(src[1], result, j, std::string(out_dir));

		std::cout << mse << std::endl;
		writing_file << mse << std::endl;

		src[0] = result;

		double percent = 0.01;


		src[0] = result(cv::Rect(static_cast<int>(result.cols*percent), 
			static_cast<int>(result.rows*percent),
			static_cast<int>(result.cols*(1.0-2*percent)), 
			static_cast<int>(result.rows*(1.0-2*percent))));

		src[1] = src[1](cv::Rect(static_cast<int>(result.cols*percent),
			static_cast<int>(result.rows*percent),
			static_cast<int>(result.cols*(1.0 - 2*percent)),
			static_cast<int>(result.rows*(1.0 - 2*percent))));

	}
}