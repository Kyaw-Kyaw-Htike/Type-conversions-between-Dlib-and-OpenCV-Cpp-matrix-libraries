#ifndef TYPEEXG_DLIB_OPENCV_H
#define TYPEEXG_DLIB_OPENCV_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "opencv2/opencv.hpp"
#include "dlib/image_transforms.h"
#include "dlib/matrix.h"
#include "dlib/array2d.h"
#include "dlib/array.h"

namespace hpers_TEDlibOpencv
{
	// convert typename and nchannels to opencv mat type such as CV_32FC1
	template <typename T>
	int getOpencvType(int nchannels)
	{
		int depth = cv::DataType<T>::depth;
		return (CV_MAT_DEPTH(depth) + (((nchannels)-1) << CV_CN_SHIFT));
	}
}

// T can be array2d<rgb_pixel>, array2d<bgr_pixel>, array2d<hsi_pixel>, array2d<lab_pixel>
// all the above cases correspond to color images with 3 channels.
// T can also be: array2d<rgb_alpha_pixel>; this has 4 channels
// it can also be: array2d<unsigned char>, array2d<int>, array2d<double>
// Then, it will be grayscale image.
// This function makes a deep copy by default. But can opt for memory sharing.
template <typename T>
void dlib2opencv(T &mat_in, cv::Mat &mat_out, bool deep_copy = true)
{
	mat_out = dlib::toMat(mat_in);
	if (deep_copy) mat_out = mat_out.clone();
}

// RGB color image
// This function makes a deep copy.
void dlib2opencv(const dlib::array2d<dlib::rgb_pixel> &mat_in, cv::Mat &mat_out)
{
	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 3;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<unsigned char>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i,j)[0] = mat_in[i][j].red;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[1] = mat_in[i][j].green;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[2] = mat_in[i][j].blue;
}

// BGR color image
void dlib2opencv(const dlib::array2d<dlib::bgr_pixel> &mat_in, cv::Mat &mat_out)
{

	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 3;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<unsigned char>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[0] = mat_in[i][j].blue;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[1] = mat_in[i][j].green;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[2] = mat_in[i][j].red;
}

// HSI color image
void dlib2opencv(const dlib::array2d<dlib::hsi_pixel> &mat_in, cv::Mat &mat_out)
{

	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 3;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<unsigned char>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[0] = mat_in[i][j].h;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[1] = mat_in[i][j].s;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[2] = mat_in[i][j].i;
}

// LAB color image
void dlib2opencv(const dlib::array2d<dlib::lab_pixel> &mat_in, cv::Mat &mat_out)
{

	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 3;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<unsigned char>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[0] = mat_in[i][j].l;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[1] = mat_in[i][j].a;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[2] = mat_in[i][j].b;
}


// RGBA color image
// This function makes a deep copy.
void dlib2opencv(const dlib::array2d<dlib::rgb_alpha_pixel> &mat_in, cv::Mat &mat_out)
{
	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 4;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<unsigned char>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[0] = mat_in[i][j].red;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[1] = mat_in[i][j].green;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[2] = mat_in[i][j].blue;

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<cv::Vec<unsigned char, nchannels>>(i, j)[3] = mat_in[i][j].alpha;
}


// T can be unsigned char, unsigned short, unsigned int, unsigned long,
// char, signed char, short, int, long
template <typename T>
void dlib2opencv(const dlib::array2d<T> &mat_in, cv::Mat &mat_out)
{
	int nrows = dlib::num_rows(mat_in);
	int ncols = dlib::num_columns(mat_in);
	const int nchannels = 1;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<T>(i, j) = mat_in[i][j];
}

// T should be generic C++ native types
// the 2nd arg mat_out will always be a 2D matrix with single channel
// this function makes a deep copy
template <typename T>
void dlib2opencv(const dlib::matrix<T> &mat_in, cv::Mat &mat_out)
{
	//deep copy
	int nr = mat_in.nr();
	int nc = mat_in.nc();

	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nc; j++)
			mat_out.at<T>(i, j) = mat_in(i, j);
}

// T should be generic C++ native types
// the 2nd arg mat_out will always be a column vector
// this function makes a deep copy
template <typename T>
void dlib2opencv(const dlib::matrix<T, 0, 1> &mat_in, cv::Mat &mat_out)
{
	int nrows = mat_in.nr();
	//int ncols = mat_in.nc();
	int ncols = 1;
	const int nchannels = 1;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));
	
	for (unsigned int i = 0; i < nrows; i++)
		mat_out.at<T>(i, 0) = mat_in(i);
}

// T should be generic C++ native types
// the 2nd arg mat_out will always be a row vector
// this function makes a deep copy
template <typename T>
void dlib2opencv(const dlib::matrix<T, 1, 0> &mat_in, cv::Mat &mat_out)
{
	//int nrows = mat_in.nr();
	int nrows = 1;
	int ncols = mat_in.nc();
	const int nchannels = 1;

	mat_out.create(nrows, ncols, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));
	
	for (unsigned int i = 0; i < ncols; i++)
		mat_out.at<T>(0, i) = mat_in(i);
}


// T should be generic C++ native types
// this function makes a deep copy
template <typename T, int nchannels>
void dlib2opencv(const dlib::array2d<dlib::matrix<T, nchannels, 1L>> &mat_in, cv::Mat &mat_out)
{
	//deep copy
	int nr = mat_in.nr();
	int nc = mat_in.nc();
	//int nchannels = mat_in[0][0].nr() * mat_in[0][0].nc();

	dlib::matrix<T, nchannels, 1L> m;
	cv::Vec<T, nchannels> v;	
	
	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nc; j++)
		{		
			m = mat_in[i][j];
			for (int k = 0; k < nchannels; k++)
			{
				v[k] = m(k);
			}
			mat_out.at<cv::Vec<T, nchannels>>(i, j) = v;
		}				
}

// T should be generic C++ native types
// this function makes a deep copy
template <typename T, int nchannels>
void dlib2opencv(const dlib::array<dlib::array2d<T>> &mat_in, cv::Mat &mat_out)
{
	int nr = mat_in[0].nr();
	int nc = mat_in[0].nc();
	//int nchannels = mat_in.array_size();

	cv::Vec<T, nchannels> v;

	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nc; j++)
		{
			for (int k = 0; k < nchannels; k++)
			{
				v[k] = mat_in[k][i][j];
			}
			mat_out.at<cv::Vec<T, nchannels>>(i, j) = v;
		}
}

// T should be generic C++ native types
template <typename T, int nchannels>
void dlib2opencv(const dlib::array<dlib::matrix<T>> &mat_in, cv::Mat &mat_out)
{
	//deep copy
	int nr = mat_in[0].nr();
	int nc = mat_in[0].nc();
	//int nchannels = mat_in.array_size();

	cv::Vec<T, nchannels> v;

	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nc; j++)
		{
			for (int k = 0; k < nchannels; k++)
			{
				v[k] = mat_in[k](i,j);
			}
			mat_out.at<cv::Vec<T, nchannels>>(i, j) = v;
		}
}

// T should be generic C++ native types
template <typename T, int nchannels>
void dlib2opencv(const std::vector<dlib::matrix<T>> &mat_in, cv::Mat &mat_out)
{
	int nr = mat_in[0].nr();
	int nc = mat_in[0].nc();
	//int nchannels = mat_in.size();

	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	dlib::matrix<T> t;

	for (int k = 0; k < nchannels; k++)
	{
		t = mat_in[k];
		for (int j = 0; j < nc; j++)
			for (int i = 0; i < nr; i++)
				mat_out.at<cv::Vec<T, nchannels>>(i, j)[k] = t(i, j);
	}
}

// T should be generic C++ native types
template <typename T, int nrows, int ncols, int nchannels>
void dlib2opencv(const std::vector<dlib::matrix<T, nrows, ncols>> &mat_in, cv::Mat &mat_out)
{
	//int nr = mat_in[0].nr();
	//int nc = mat_in[0].nc();
	//int nchannels = mat_in.size();
	int nr = nrows;
	int nc = ncols;	

	mat_out.create(nr, nc, hpers_TEDlibOpencv::getOpencvType<T>(nchannels));

	dlib::matrix<T, nrows, ncols> t;

	for (int k = 0; k < nchannels; k++)
	{
		t = mat_in[k];
		for (int j = 0; j < nc; j++)
			for (int i = 0; i < nr; i++)
				mat_out.at<cv::Vec<T, nchannels>>(i, j)[k] = t(i, j);
	}
}


// T can be rgb_pixel, bgr_pixel, hsi_pixel, lab_pixel
// all the above cases correspond to color images with 3 channels (which mat_in should be)
// T can also be: rgb_alpha_pixel; this corresponds to 4 channels
// it can also be: unsigned char, int, double
// Then, mat_in is assumd to be a grayscale image.
// mat_out is an object of class cv_image<T> and it is very similar to array2d<T>
// This function does NOT make a DEEP copy.
template <typename T>
void opencv2dlib(cv::Mat mat_in, dlib::cv_image<T> &mat_out)
{
	mat_out = dlib::cv_image<T>(mat_in);
}

// RGB color image
// This function makes a deep copy.
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::rgb_pixel> &mat_out)
{

	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	const int nchannels = 3;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].red = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[0];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].green = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[1];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].blue = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[2];
}

// BGR color image
// This function makes a deep copy.
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::bgr_pixel> &mat_out)
{

	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	const int nchannels = 3;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].blue = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[0];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].green = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[1];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].red = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[2];
}

// HSI color image
// This function makes a deep copy.
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::hsi_pixel> &mat_out)
{

	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	const int nchannels = 3;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].h = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[0];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].s = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[1];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].i = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[2];
}

// LAB color image
// This function makes a deep copy.
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::lab_pixel> &mat_out)
{

	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	const int nchannels = 3;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].l = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[0];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].a = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[1];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].b = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[2];
}

// RGBA color image
// This function makes a deep copy.
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::rgb_alpha_pixel> &mat_out)
{

	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	const int nchannels = 4;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].red = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[0];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].green = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[1];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].blue = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[2];

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out[i][j].alpha = mat_in.at<cv::Vec<unsigned char, nchannels>>(i, j)[3];
}

// T can be rgb_pixel, bgr_pixel, hsi_pixel, lab_pixel
// all the above cases correspond to color images with 3 channels (which mat_in should be)
// T can also be: rgb_alpha_pixel; this corresponds to 4 channels
// it can also be: unsigned char, int, double
// Then, mat_in is assumd to be a grayscale image.
// mat_out is an object of class cv_image<T> and it is very similar to array2d<T>
// This function makes a DEEP copy.
template <typename T>
void opencv2dlib(cv::Mat mat_in, dlib::array2d<T> &mat_out)
{
	dlib::assign_image(mat_out, dlib::cv_image<T>(mat_in));
}

// T should be generic C++ native types
// this is obviously only for 2D matrices (1 channel)
// makes deep copy
template <typename T>
void opencv2dlib(const cv::Mat &mat_in, dlib::matrix<T> &mat_out)
{
	int nrows = mat_in.rows;
	int ncols = mat_in.cols;

	mat_out.set_size(nrows, ncols);

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out(i,j) = mat_in.at<T>(i,j);
}

// T should be generic C++ native types
// the 2nd arg mat_out will always be a column vector
// this function makes a deep copy
template <typename T>
void opencv2dlib(const cv::Mat &mat_in, dlib::matrix<T, 0, 1> &mat_out)
{

	int nrows = mat_in.rows;
	//int ncols = mat_in.cols;
	int ncols = 1;

	mat_out.set_size(nrows);

	for (unsigned int i = 0; i < nrows; i++)
		mat_out(i) = mat_in.at<T>(i,0);
}

// T should be generic C++ native types
// the 2nd arg mat_out will always be a row vector
// this function makes a deep copy
template <typename T>
void opencv2dlib(const cv::Mat &mat_in, dlib::matrix<T, 1, 0> &mat_out)
{
	//int nrows = mat_in.rows;
	int ncols = mat_in.cols;

	mat_out.set_size(ncols);

	for (unsigned int i = 0; i < ncols; i++)
		mat_out(i) = mat_in.at<T>(0, i);
}

// T should be generic C++ native types
// mat_in (the 1st arg) should be either only one row or one column (i.e. similar to a vector)
// this function makes a deep copy
template <typename T, int nchannels>
void opencv2dlib(const cv::Mat &mat_in, dlib::array2d<dlib::matrix<T, nchannels, 1L>> &mat_out)
{
	int nr = mat_in.rows;
	int nc = mat_in.cols;
	//int nchannels = mat_in.channels();

	mat_out.set_size(nr, nc);

	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < nc; j++)
			for (int i = 0; i < nr; i++)
				mat_out[i][j](k) = mat_in.at<cv::Vec<T,nchannels>>(i,j)[k];
}

// T should be generic C++ native types
// this function makes a deep copy
template <typename T, int nchannels>
void opencv2dlib(const cv::Mat &mat_in, dlib::array<dlib::array2d<T>> &mat_out)
{
	int nr = mat_in.rows;
	int nc = mat_in.cols;
	//int nchannels = mat_in.channels();

	mat_out.set_max_size(nchannels);
	mat_out.set_size(nchannels);

	for (int k = 0; k < nchannels; k++)
		mat_out[k].set_size(nr, nc);


	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < nc; j++)
			for (int i = 0; i < nr; i++)
				mat_out[k][i][j] = mat_in.at<cv::Vec<T, nchannels>>(i, j)[k];
}

// T should be generic C++ native types
template <typename T, int nchannels>
void opencv2dlib(const cv::Mat &mat_in, dlib::array<dlib::matrix<T>> &mat_out)
{
	int nr = mat_in.rows;
	int nc = mat_in.cols;
	//int nchannels = mat_in.channels();

	mat_out.set_max_size(nchannels);
	mat_out.set_size(nchannels);

	dlib::matrix<T> temp(nr, nc);

	for (int k = 0; k < nchannels; k++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int i = 0; i < nr; i++)
				temp(i, j) = mat_in.at<cv::Vec<T, nchannels>>(i, j)[k];
		}
		mat_out[k] = temp;
	}
}

// T should be generic C++ native types
template <typename T, int nchannels>
void opencv2dlib(const cv::Mat &mat_in, std::vector<dlib::matrix<T>> &mat_out)
{
	int nr = mat_in.rows;
	int nc = mat_in.cols;
	//int nchannels = mat_in.channels();

	mat_out.resize(nchannels);

	dlib::matrix<T> temp(nr, nc);

	for (int k = 0; k < nchannels; k++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int i = 0; i < nr; i++)
				temp(i, j) = mat_in.at<cv::Vec<T, nchannels>>(i, j)[k];
		}
		mat_out[k] = temp;
	}
}

// T should be generic C++ native types
template <typename T, int nrows, int ncols, int nchannels>
void opencv2dlib(const cv::Mat &mat_in, std::vector<dlib::matrix<T, nrows, ncols>> &mat_out)
{
	//int nr = mat_in.rows;
	//int nc = mat_in.cols;
	//int nchannels = mat_in.channels();
	int nr = nrows;
	int nc = ncols;

	mat_out.resize(nchannels);

	dlib::matrix<T, nrows, ncols> temp;

	for (int k = 0; k < nchannels; k++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int i = 0; i < nr; i++)
				temp(i, j) = mat_in.at<cv::Vec<T, nchannels>>(i, j)[k];
		}
		mat_out[k] = temp;
	}
}



#endif
