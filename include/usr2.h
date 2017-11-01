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