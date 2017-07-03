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
#include <cmdline.h>

#include "convert.h"

double mean_squeare_error(cv::Mat origin, cv::Mat edited, std::string out_dir) {
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
	cv::imwrite(out_dir + "\\diff" + ".jpg", gray8u_o);
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

int main() {
	std::cout << "Start.\n";
	using cv::imread;
	using cv::Mat;
	using std::vector;
	using cv::imshow;
	using cv::waitKey;
	using cv::Size;
	using cv::Vec3b;

	Mat src[2];

	//std::string origin_name = "images/sample.jpg";
	//std::string photo_name = "images/sample_withnotes.jpg";

	std::string origin_name = "images/656-origin.jpg";
	std::string photo_name = "images/656-top-notes.jpg";

	//std::string origin_name = "lena.jpg";
	//std::string photo_name = "printed_lena.jpg";

	// 結果保存用のフォルダとファイル
	//char out_dir[] = "";
	char out_dir[] = "test";
	//time_t now = time(NULL);
	//struct tm *pnow = localtime(&now);
	//sprintf(out_dir, "results/%s%02d%02d%02d%02d",
	//	split(photo_name, '.')[0].c_str(), pnow->tm_mon + 1, pnow->tm_mday, pnow->tm_hour, pnow->tm_min);
	_mkdir(out_dir);

	std::cout << std::string(out_dir) + "\n";
	//std::cout << "Start reading images.\n";

	src[0] = imread(origin_name);
	src[1] = imread(photo_name);

	std::cout << "Finished resizing images.\n";
	std::cout << "Start resizing images.\n";

	std::cout << "Finished resizing images.\n";
	std::cout << "Start converting images.\n";

	// gray 画像にする
	//gray[0].copyTo(src[0]);

	Mat result;
	float reduction_rate = 0.5;
	convertImage(src[0], src[1], result, reduction_rate);
	Mat notes = result.clone();
	extract_notes(result, notes);
	Mat withnotes = src[0].clone();
	add_notes(src[0], notes, withnotes);

	imshow("result img", result);
	imshow("notes img", notes);
	imshow("withnotes img", withnotes);

	cv::imwrite(std::string(out_dir) + "/" + "result.jpg", result);
	cv::imwrite(std::string(out_dir) + "/" + "notes.jpg", notes);
	cv::imwrite(std::string(out_dir) + "/" + "withnotes.jpg", withnotes);
	waitKey(0);

	std::cout << "Finished resizing images.\n";
	return 0;
}