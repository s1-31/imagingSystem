#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>

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

	src[0] = imread("lena.jpg");
	src[1] = imread("printed_lena.jpg");
	
	cv::cvtColor(src[0], gray[0], CV_BGR2GRAY);
	cv::cvtColor(src[1], gray[1], CV_BGR2GRAY);

	// gray ‰æ‘œ‚É‚·‚é
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
	waitKey();



	Mat homo = cv::findHomography(points1, points2, CV_RANSAC);
	cv::warpPerspective(src[0], result, homo, Size(static_cast<int>(src[1].cols), static_cast<int>(src[1].rows)));
	waitKey();
	for (int y = 0; y < src[0].rows; y++) {
		for (int x = 0; x < src[0].cols; x++) {
			result.at<Vec3b>(y, x) = src[1].at<Vec3b>(y, x);
		}
	}

	imshow("result img", result);
	waitKey(0);

	Mat result2;

	Mat homo2 = cv::findHomography(points2, points1, CV_RANSAC);
	cv::warpPerspective(result, result2, homo2, Size(static_cast<int>(src[1].cols), static_cast<int>(src[1].rows)));

	imshow("warped result", result2);
	waitKey(0);

	for (int y = 0; y < src[0].rows; y++) {
		for (int x = 0; x < src[0].cols; x++) {
			result2.at<Vec3b>(y, x) -= src[0].at<Vec3b>(y, x);
		}
	}

	cv::Mat crop = result2(cv::Rect(0, 0, src[0].cols, src[0].rows));

	imshow("substracttion", crop);
	waitKey(0);
}