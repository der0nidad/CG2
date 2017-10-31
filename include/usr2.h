#pragma once

#include "matrix.h"
#include "EasyBMP.h"

#include <tuple>

typedef Matrix<std::tuple<uint, uint, uint>> Image;

Image load_image(const char*);
void save_image(const Image&, const char*);

Image Sobel_X_Img(Image & image);
Matrix<float> Sobel_Y_Img(Image & image);