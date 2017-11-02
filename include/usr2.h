#pragma once

#include "matrix.h"
#include "EasyBMP.h"

#include <tuple>

typedef Matrix<std::tuple<uint, uint, uint>> Image;

Image load_image(const char*);
void save_image(const Image&, const char*);

Matrix<float> Sobel_Y_Img(Matrix<float> & image);

Matrix<float> Sobel_X_Img(Matrix<float> & image);

void save_Matrix(const Matrix<float> &im, const char *path);

Matrix<float> grayscale_Matrix_from_BMP(BMP & in);

Matrix<float> SobelConvolutionImg(Matrix<float> & image_X, Matrix<float> & image_Y);

Matrix<float> SobelConvolutionImg2(Matrix<float>& image, Matrix<float> &resAngle);

std::vector<float> Calc_Histogram(Matrix<float>& module, Matrix<float>& angle);

std::vector<float> Calc_Cell_Hist(Matrix<float>& module,Matrix<float>& angle, uint begHeight, uint begWidth, uint endHeight, uint endWidth);

std::pair<int,int> Calc_Max_Wid_Heig(const std::vector<std::pair<BMP*, int> > & data_set);