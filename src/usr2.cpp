 // #include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "EasyBMP.h"
#include "matrix.h"
#include "usr2.h"

using std::shared_ptr;
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::get;
using std::tuple;
using std::make_tuple;
using std::tie;


/*

Image Sobel_X_Img(Image & image){
         std::vector<int> pixel(4);

     Image result(image.n_rows, image.n_cols);
    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, 0, 1,*вторая строка*\/ -2, 0, 2,*третья строка*//* -1, 0, 1} ;    

    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {
            for (uint k = 0; k < 3; ++k)
            {
                for (uint l = 0; l < 3; ++l)
                {
                    pixel.at(chnl) += get<0> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 1) += get<1> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 2) += get<2> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                }
            }
            // pixel.at(chnl) /= 6;
            // pixel.at(chnl + 2) /= 6;
            // pixel.at(chnl + 1) /= 6;
            
            if(pixel.at(chnl) > 255) {pixel.at(chnl) = 255;}
            // if(pixel.at(chnl + 1) > 255) {pixel.at(chnl + 1) = 255;}
            // if(pixel.at(chnl + 2) > 255) {pixel.at(chnl + 2) = 255;}

            if(pixel.at(chnl) < 0) {pixel.at(chnl) = 0;}
            // if(pixel.at(chnl + 1) < 0 ) {pixel.at(chnl + 1) = 0;}
            // if(pixel.at(chnl + 2) < 0) {pixel.at(chnl + 2) = 0;}

            result(i,j) = make_tuple(pixel.at(0), pixel.at(0), pixel.at(0));
        }
    }
    return result;
}*/
Matrix<float> Sobel_Y_Img(Matrix<float> & image){
    /*typedef std::vector<int> Vec;
...
Vec* pVec = new Vec;
или
shared_ptr<Vec> pVec = shared_ptr<Vec>(new Vec);
 
.....
delete pVec;*/
     Matrix<float> result(image.n_rows, image.n_cols);
    // std::vector<int> *pixel = new std::vector<int> ;
    // shared_ptr<vector<int>> pVec = shared_ptr<vector<int>>(new vector<int>);
    std::vector<int> pixel (2) ;

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, -2, -1,/*вторая строка*/ 0, 0, 0,/*третья строка*/ 1, 2, 1} ;

    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {
            for (uint k = 0; k < 3; ++k)
            {
                for (uint l = 0; l < 3; ++l)
                {
                    pixel.at(chnl) += (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 1) += get<1> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 2) += get<2> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                }
            }
            // pixel.at(chnl) /= 6;
            // pixel.at(chnl + 2) /= 6;
            // pixel.at(chnl + 1) /= 6;
            
            if(pixel.at(chnl) > 255) {pixel.at(chnl) = 255;}
            // if(pixel.at(chnl + 1) > 255) {pixel.at(chnl + 1) = 255;}
            // if(pixel.at(chnl + 2) > 255) {pixel.at(chnl + 2) = 255;}

            if(pixel.at(chnl) < 0) {pixel.at(chnl) = 0;}
            // if(pixel.at(chnl + 1) < 0 ) {pixel.at(chnl + 1) = 0;}
            // if(pixel.at(chnl + 2) < 0) {pixel.at(chnl + 2) = 0;}

            result(i,j) = pixel.at(0);
        }
    }
    return result;
}


Matrix<float> Sobel_X_Img(Matrix<float> & image){
    /*typedef std::vector<int> Vec;
...
Vec* pVec = new Vec;
или
shared_ptr<Vec> pVec = shared_ptr<Vec>(new Vec);
 
.....
delete pVec;*/
     Matrix<float> result(image.n_rows, image.n_cols);
    // std::vector<int> *pixel = new std::vector<int> ;
    // shared_ptr<vector<int>> pVec = shared_ptr<vector<int>>(new vector<int>);
    std::vector<int> pixel (2) ;

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, 0, 1,/*вторая строка*/ -2, 0, 2,/*третья строка*/ -1, 0, 1} ;    

    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {
            for (uint k = 0; k < 3; ++k)
            {
                for (uint l = 0; l < 3; ++l)
                {
                    pixel.at(chnl) += (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 1) += get<1> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                    // pixel.at(chnl + 2) += get<2> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                }
            }
            // pixel.at(chnl) /= 6;
            // pixel.at(chnl + 2) /= 6;
            // pixel.at(chnl + 1) /= 6;
            
            if(pixel.at(chnl) > 255) {pixel.at(chnl) = 255;}
            // if(pixel.at(chnl + 1) > 255) {pixel.at(chnl + 1) = 255;}
            // if(pixel.at(chnl + 2) > 255) {pixel.at(chnl + 2) = 255;}

            if(pixel.at(chnl) < 0) {pixel.at(chnl) = 0;}
            // if(pixel.at(chnl + 1) < 0 ) {pixel.at(chnl + 1) = 0;}
            // if(pixel.at(chnl + 2) < 0) {pixel.at(chnl + 2) = 0;}

            result(i,j) = pixel.at(0);
        }
    }
    return result;
}
Matrix<float> SobelConvolutionImg(Matrix<float> & image_X, Matrix<float> & image_Y){
Matrix<float> res = Matrix<float>(image_X.n_rows, image_X.n_cols);
std::vector<float> pixel (2) ;

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, 0, 1,/*вторая строка*/ -2, 0, 2,/*третья строка*/ -1, 0, 1} ;    

    for (uint i = 1; i < image_X.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image_X.n_cols - 1; ++j)
        {


            pixel.at(chnl) = hypotf(image_X(i,j), image_Y(i,j));
            if(pixel.at(chnl) > 255) {pixel.at(chnl) = 255;}
            // if(pixel.at(chnl + 1) > 255) {pixel.at(chnl + 1) = 255;}
            // if(pixel.at(chnl + 2) > 255) {pixel.at(chnl + 2) = 255;}

            if(pixel.at(chnl) < 0) {pixel.at(chnl) = 0;}
            // if(pixel.at(chnl + 1) < 0 ) {pixel.at(chnl + 1) = 0;}
            // if(pixel.at(chnl + 2) < 0) {pixel.at(chnl + 2) = 0;}

            res(i,j) = pixel.at(chnl);
        }
    }



return res;
}
inline int calc_brightness_gray(uint b, uint g, uint r){
    double a  = r * 0.299 + g * 0.587 + b * 0.114;
    return static_cast<int>(a);
}

Image image_from_BMP(BMP & in){
    Image res(in.TellHeight(), in.TellWidth());
    for (uint i = 0; i < res.n_rows; ++i) {
        for (uint j = 0; j < res.n_cols; ++j) {
            RGBApixel *p = in(j, i);
            res(i, j) = make_tuple(p->Red, p->Green, p->Blue);
        }
    }

    return res;
}
Matrix<float> grayscale_Matrix_from_BMP(BMP & in){
    Matrix<float> res(in.TellHeight(), in.TellWidth());
    int pix = 0;
    for (uint i = 0; i < res.n_rows; ++i) {
        for (uint j = 0; j < res.n_cols; ++j) {
            RGBApixel *p = in(j, i);
            pix = calc_brightness_gray(p->Blue, p->Green, p->Red);
            p->Red = pix;
            p->Green= pix;
            p->Blue = pix;
            res(i, j) = pix;
        }
    }

    return res;
}



Image load_image(const char *path)
{
    BMP in;

    if (!in.ReadFromFile(path))
        throw string("Error reading file ") + string(path);

    Image res(in.TellHeight(), in.TellWidth());

    for (uint i = 0; i < res.n_rows; ++i) {
        for (uint j = 0; j < res.n_cols; ++j) {
            RGBApixel *p = in(j, i);
            res(i, j) = make_tuple(p->Red, p->Green, p->Blue);
        }
    }

    return res;
}

void save_image(const Image &im, const char *path)
{
    BMP out;
    out.SetSize(im.n_cols, im.n_rows);

    uint r, g, b;
    RGBApixel p;
    p.Alpha = 255;
    for (uint i = 0; i < im.n_rows; ++i) {
        for (uint j = 0; j < im.n_cols; ++j) {
            tie(r, g, b) = im(i, j);
            p.Red = r; p.Green = g; p.Blue = b;
            out.SetPixel(j, i, p);
        }
    }

    if (!out.WriteToFile(path ))
        throw string("Error writing file ") + string(path);
}
 

void save_Matrix(const Matrix<float> &im, const char *path)
{
    BMP out;
    out.SetSize(im.n_cols, im.n_rows);

    float r;
    RGBApixel p;
    p.Alpha = 255;
    for (uint i = 0; i < im.n_rows; ++i) {
        for (uint j = 0; j < im.n_cols; ++j) {
            r = im(i, j);
            p.Red = r; p.Green = r; p.Blue = r;
            out.SetPixel(j, i, p);
        }
    }

    if (!out.WriteToFile(path ))
        throw string("Error writing file ") + string(path);
}
 