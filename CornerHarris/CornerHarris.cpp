// CornerHarris.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

Mat src, src_gray;
RNG rng(12345);
auto thresh = 200;
auto max_thresh = 255;
auto maxCorners = 1000;
auto maxTrackbar = 200;
auto minHessian = 400;
void readme();
void cornerHarris_demo(int, void*);
void goodFeaturesToTrack_Demo(int, void*);

int main()
{
	Mat tmp, in, feat1, feat2, descriptor1, descriptor2, result;
	char* img1_file = "grosssued.png";
	char* img2_file = "2_b_10_0002800__1.bmp";
	char* sift = "SIFT";
	tmp = imread(img1_file, 1);
	in = imread(img2_file, 1);
	if (in.empty() || tmp.empty())
		cout << "failed to open img.jpg" << endl;
	else
		cout << "img.jpg loaded OK" << endl;
	// SIFT feature detector and feature extractor  /* threshold      = 0.04;  edge_threshold = 10.0;  magnification  = 3.0;    */
	SiftFeatureDetector detector(400, 5);
	SiftDescriptorExtractor extractor(3);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(tmp, keypoints1);
	detector.detect(in, keypoints2);
	drawKeypoints(tmp, keypoints1, feat1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(in, keypoints2, feat2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imwrite("feat1.bmp", feat1);
	imwrite("feat2.bmp", feat2);
	int key1 = keypoints1.size();
	int key2 = keypoints2.size();
	printf("Keypoint1=%d \nKeypoint2=%d", key1, key2);
	extractor.compute(tmp, keypoints1, descriptor1);
	extractor.compute(in, keypoints2, descriptor2);
	vector<DMatch> matches;
	BruteForceMatcher<L2<float>> matcher;
	matcher.match(descriptor1, descriptor2, matches);
	Mat;
	drawMatches(tmp, keypoints1, in, keypoints2, matches, result);
	imwrite("result.bmp", result);
	namedWindow(sift, CV_WINDOW_AUTOSIZE);
	resizeWindow(sift, 800, 800);
	imshow(sift, result);
	waitKey(0);
	tmp.refcount = nullptr;
	in.refcount = nullptr;
	feat1.refcount = nullptr;
	feat2.refcount = nullptr;
	descriptor1.refcount = nullptr;
	descriptor2.refcount = nullptr;
	result.refcount = nullptr;
	tmp.release();
	in.release();
	feat1.release();
	feat2.release();
	descriptor1.release();
	descriptor2.release();
	result.release();
	return 0;
}

void cornerHarris_demo(int, void*)
{
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	char* corners_window = "Corners Detected";

	//Detector parameters
	int blockSize = 5;
	int apertureSize = 9;
	double k = 0.05;

	//Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	//Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	//Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int) dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0, 255, 255), 2, 8, 0);
			}
		}
	}
	namedWindow(corners_window, 0);
	resizeWindow(corners_window, 800, 800);
	imshow(corners_window, dst_norm_scaled);
	dst.refcount = nullptr;
	dst_norm.refcount = nullptr;
	dst_norm_scaled.refcount = nullptr;
	dst.release();
	dst_norm.release();
	dst_norm_scaled.release();
}

void goodFeaturesToTrack_Demo(int, void*)
{
	char* goodfeatures_window = "Goodfeatures";
	if (maxCorners < 1)
		maxCorners = 1;

	//Parameters for Shi-Tomasi algorithm
	vector<Point2f> corners;
	double qualityLevel = 0.1;
	double minDistance = 5;
	int blockSize = 5;
	bool useHarrisDetector = false;
	double k = 0.04;

	// Copy the source image
	Mat copy;
	copy = src.clone();

	//Apply corner detection
	goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

	//Draw corners detected
	cout << "** Number of corners detected: " << corners.size() << endl;
	int r = 15;
	for (int i = 0; i < corners.size(); i++)
	{
		circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 10, 8, 0);
	}

	namedWindow(goodfeatures_window, 0);
	resizeWindow(goodfeatures_window, 800, 800);
	imshow(goodfeatures_window, copy);
	copy.refcount = nullptr;
	copy.release();
}

void readme()
{
	cout << " Usage: ./SURF_detector <img1> " << endl; }
