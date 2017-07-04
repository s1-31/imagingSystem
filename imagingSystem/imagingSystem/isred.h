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

#include "filter.h"


bool isred_basedon_global_std(int*, double*, double*);
bool isred_basedon_pstd(int, int, int);
bool isred(cv::Vec3b);
bool isred(cv::Vec3b, double*, double*);
bool isred_basedon_edges(int, int, int);
bool isred_basedon_edges(int*, double*, double*);


bool isred_basedon_pstd(int r, int b, int g) {
	double threshold = 0.02;
	double vlth = 0.1;

	double db = b / 255.0;
	double dg = g / 255.0;
	double dr = r / 255.0;

	double m = (dr + dg + db) / 3.0;
	double norm_std = sqrt((db - m)*(db - m) + (dg - m)*(dg - m) + (dr - m)*(dr - m)) / m;
	if (norm_std > threshold && dr > dg && dr > db && (norm_std*m) > vlth) {
		return true;
	}
	return false;
}

bool isred_basedon_global_std(int* p, double* gm, double* gstd) {
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


bool isred(cv::Vec3b pixel) {
	int b = pixel(0);
	int g = pixel(1);
	int r = pixel(2);

	//return isred_based_pstd(r, g, b);
	return isred_basedon_edges(r, g, b);
}

bool isred(cv::Vec3b pixel, double* gm, double* gstd) {
	int b = pixel(0);
	int g = pixel(1);
	int r = pixel(2);
	int p[] = { r, g, b };

	//return isred_based_global_std(p, gm, gstd);
	return isred_basedon_edges(p, gm, gstd);
}

bool isred_basedon_edges(int r, int g, int b) {

	double threshold = 0.2;

	double rg = (r-g) / 255.0;
	double gb = (g-b) / 255.0;
	double br = (b-r) / 255.0;

	double norm_std = sqrt(rg*rg + gb*gb + br*br);
	if (norm_std > threshold) {
		return true;
	}
	return false;
}

bool isred_basedon_edges(int* p, double* gm, double* gstd) {
	//double threshold = 4.5;
	double threshold = 10.0;

	double rg = (p[0] - p[1]) / 255.0;
	double gb = (p[1] - p[2]) / 255.0;
	double br = (p[2] - p[0]) / 255.0;

	double norm_rg = (rg - gm[0]) / gstd[0];
	double norm_gb = (gb - gm[1]) / gstd[1];
	double norm_br = (br - gm[2]) / gstd[2];

	//if (fabs(norm_rg) > threshold || fabs(norm_gb) > threshold || fabs(norm_br) > threshold) {
	if (sqrt(norm_rg*norm_rg + norm_gb*norm_gb + norm_br*norm_br) > threshold) {
		return true;
	}
	return false;
}
