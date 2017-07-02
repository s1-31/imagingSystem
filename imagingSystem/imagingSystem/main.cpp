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

bool pixel_based_std(int r, int b, int g) {
	double threshold = 0.03;

	double db = b / 255.0;
	double dg = g / 255.0;
	double dr = r / 255.0;

	double m = (dr + dg + db) / 3.0;
	double std = ((db - m)*(db - m) + (dg - m)*(dg - m) + (dr - m)*(dr - m)) / m;
	if (std > threshold && dr > dg && dr > db) {
		return true;
	}
	return false;
}

bool isred(cv::Vec3b pixel) {
	int b = pixel(0);
	int g= pixel(1);
	int r= pixel(2);

	return pixel_based_std(r, g, b);
}

void extract_notes(cv::Mat image, cv::Mat& notes) {
	std::cout << "Start Extracting notes.";
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (isred(image.at<cv::Vec3b>(y, x))) {
				notes.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
			}
			else {
				// notes.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
				notes.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}
		}
	}
}

void add_notes(cv::Mat original, cv::Mat notes, cv::Mat& withnotes) {
	for (int y = 0; y < withnotes.rows; y++) {
		for (int x = 0; x < withnotes.cols; x++) {
			// cv::add(original.at<cv::Vec3b>(y, x), notes.at<cv::Vec3b>(y, x), withnotes.at<cv::Vec3b>(y, x));
			if (notes.at<cv::Vec3b>(y, x)(2) > 0) {
				withnotes.at<cv::Vec3b>(y, x) = notes.at<cv::Vec3b>(y, x);
			}
			else {
				withnotes.at<cv::Vec3b>(y, x) = original.at<cv::Vec3b>(y, x);
			}
		}
	}
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
	Mat gray[2];

	std::string origin_name = "images/sample.jpg";
	std::string photo_name = "images/sample_withnotes.jpg";

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

	std::cout << "Start reading images.\n";


	src[0] = imread(origin_name);
	src[1] = imread(photo_name);

	std::cout << "Finished resizing images.\n";
	std::cout << "Start resizing images.\n";

	// temporal countermeasure
	cv::resize(src[0], src[0], cv::Size(), 0.5, 0.5);
	cv::resize(src[1], src[1], cv::Size(), 0.5, 0.5);

	std::cout << "Finished resizing images.\n";
	std::cout << "Start converting images.\n";

	cv::cvtColor(src[0], gray[0], CV_BGR2GRAY);
	cv::cvtColor(src[1], gray[1], CV_BGR2GRAY);

	// gray 画像にする
	//gray[0].copyTo(src[0]);

	Mat result;
	convertImage(src[0], src[1], result);
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