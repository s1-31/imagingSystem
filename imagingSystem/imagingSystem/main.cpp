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

#include "cmdline.h"
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

int main(int argc, char *argv[]) {
	std::cout << "Start.\n";
	using cv::imread;
	using cv::Mat;
	using std::vector;
	using cv::imshow;
	using cv::waitKey;
	using cv::Size;
	using cv::Vec3b;


	cmdline::parser parser;
	parser.add<std::string>("input", 'i', "original input image", true, "");
	parser.add<std::string>("withnotes", 'w', "photo with notes", true, "");
	parser.add<std::string>("outdir", 'd', "output directory path", false, "test");
	parser.add<std::string>("output", 'o', "output file name", false, "");
	parser.parse_check(argc, argv);

	std::string origin_name = parser.get<std::string>("input");
	std::string photo_name = parser.get<std::string>("withnotes");
	std::string out_dir = parser.get<std::string>("outdir");
	std::string origin_file_name = split(origin_name, '/').back();
	std::string out_name = parser.get<std::string>("output") == "" ? split(origin_file_name, '.')[0] + "-withnotes.jpg" : parser.get<std::string>("output");

	std::cout << out_name << std::endl;

	Mat src[2];

	src[0] = imread(origin_name);
	src[1] = imread(photo_name);

	std::cout << "Finished resizing images.\n";
	std::cout << "Start resizing images.\n";

	std::cout << "Finished resizing images.\n";
	std::cout << "Start converting images.\n";

	// gray 画像にする
	//gray[0].copyTo(src[0]);

	Mat warped;
	float reduction_rate = 0.5;
	convertImage(src[0], src[1], warped, reduction_rate);
	Mat notes = warped.clone();
	extract_notes(warped, notes);
	Mat withnotes = src[0].clone();
	add_notes(src[0], notes, withnotes);

	imshow("warped img", warped);
	imshow("notes img", notes);
	imshow("withnotes img", withnotes);

	_mkdir(out_dir.c_str());
	cv::imwrite(out_dir + "/" + split(origin_file_name, '.')[0] + "-warped.jpg", warped);
	cv::imwrite(out_dir + "/" + split(origin_file_name, '.')[0] + "-notes.jpg", notes);
	cv::imwrite(out_dir + "/" + out_name, withnotes);
	waitKey(0);

	return 0;
}