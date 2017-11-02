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

// число сегментов в гистограмме
#define NUMB_OF_SEG 8 



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

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, -2, -1,/*вторая строка*/ 0, 0, 0,/*третья строка*/ 1, 2, 1} ;

    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {
    std::vector<int> pixel (2) ;
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

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, 0, 1,/*вторая строка*/ -2, 0, 2,/*третья строка*/ -1, 0, 1} ;    

    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {
    std::vector<int> pixel (2) ;
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

    // Image kernel(3,3);
    int chnl = 0;
    std::vector<int> kernel = {-1, 0, 1,/*вторая строка*/ -2, 0, 2,/*третья строка*/ -1, 0, 1} ;    

    for (uint i = 1; i < image_X.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image_X.n_cols - 1; ++j)
        {
std::vector<float> pixel (2) ;


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

// первый элемент - ширина, второй высота - максимальные, котрые встречаются в датасете
std::pair<int,int> Calc_Max_Wid_Heig(const vector<pair<BMP*, int> > & data_set){
    std::pair<int,int> res;
    res.first = 0;
    res.second = 0;
    int maxHeight = 0, maxWidth = 0, curr_height = 0, curr_width = 0;
    int countW = 0, countH = 0;
    countW = countW;
        for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {
         curr_width = (data_set.at(image_idx).first)->TellWidth();
         curr_height= data_set.at(image_idx).first->TellHeight();
        if(curr_height >= maxHeight){
             maxHeight = curr_height;  countH = image_idx;

        }
        curr_width> maxWidth ? maxWidth= curr_width: 1;
        if(curr_width>= maxWidth){
            maxWidth= curr_width;
            countW = image_idx;
        }
    }
    cout << "номер на кот макс ширина " <<  countW << "\n";
    cout << "номер на кот макс высота " <<  countH << "\n";
    res.first = maxWidth;
    res.second = maxHeight;

    return res;
}
//--------------------------------------------------------------------------------------
// возвращаем гистограмму градиентов(массив) для одной картинки
std::vector<float> Calc_Histogram(Matrix<float>& module, Matrix<float>& angle){
    cout << "ГОМООМОМОМОМГО" << NUMB_OF_SEG << "  !";
    std::vector<float> hist0;
    int size = NUMB_OF_SEG;//кол-во сегментов
    int seg = 0/*0..7*/, cell_size_H = 0, cell_size_W= 0 ;
    cell_size_W = module.n_cols / 8;
    cell_size_H = module.n_rows / 8;
    cell_size_H = cell_size_H;
    cell_size_W = cell_size_W;
    seg = seg;
    float sum =0.f ;
    float angleOne = 360.f / size;
    angleOne = angleOne;

    angle(1,2) = 1;
    for(int hei = 0; hei < 8 ; hei++){
        for (int wid = 0; wid < 8; wid++)
        {
    std::vector<float> hist(NUMB_OF_SEG, 0);
            // cout << ++mm << "  \n";
            hist = Calc_Cell_Hist(module, angle, cell_size_H * hei, cell_size_W * wid , cell_size_H * (hei + 1),  cell_size_W * (wid + 1));
            for (int i = 0; i < NUMB_OF_SEG; ++i)
            {
                cout << "ДО НОРМИРОВАНИЯ сегмент " << i << "  значенеи " << hist.at(i) << "\n"    ;
                // нормализуем здесь
                // hist.at(i) = hist.at(i) * hist.at(i);
                sum += hist.at(i)  * hist.at(i);
                // hist0.push_back(hist.at(i));
            }
            sum = sqrt(sum);
            for (int i = 0; i < NUMB_OF_SEG; ++i)
            {
                // нормализуем здесь
                hist.at(i) =( hist.at(i) / sum);
                hist0.push_back(hist.at(i));
                cout << "ПОСЛЕ НОРМИРОВАНИЯ сегмент " << i << "  значенеи " << hist.at(i) << "\n"    ;

            }
        }
    }
    cout << "КОЛИЧЕСТВО ЭЛЕМЕНТОВ В HIST0  " << hist0.size() << "  " ;


    return hist0;
}

    std::vector<float> Calc_Cell_Hist(Matrix<float>& module,Matrix<float>& angle, uint begHeight, uint begWidth, uint endHeight, uint endWidth){
    int size = NUMB_OF_SEG;//кол-во сегментов
    std::vector<float> hist(NUMB_OF_SEG,0);
    // uint cell_width = module.n_cols / 8;
    // uint cell_height  = module.n_rows / 8;
    int seg = 0;//0..7
    seg = seg;
    // int mm =0;
    float angleOne = 360.f / size;
    for (uint i = begHeight + 1; i < endHeight - 1; i++)
    {
        for (uint j= begWidth + 1; j < endWidth - 1; j++)
        {
                    seg = angle(i, j);
                    seg /= angleOne;
                    seg+=( NUMB_OF_SEG -1) / 2;//все, вот так правильно!
            // cout << seg << "@  ";
            if(abs(seg) > 7) {cout << seg << " - вот оно ; \n";}else{
                    hist.at(seg) += module(i,j);}
                    // cout << " seg " << seg << "\n";
        }
    }
    cout << "Я ДОБЕРУСЬ ДО ТЕБЯ!";
    return hist;
}


Matrix<float> SobelConvolutionImg2(Matrix<float>& image, Matrix<float> &resAngle){
// Matrix<float> res = Matrix<float>(image.n_rows, image.n_cols);
// Matrix<float> res_Y = Matrix<float>(image.n_rows, image.n_cols);
Matrix<float> resModule = Matrix<float>(image.n_rows, image.n_cols);
// Matrix<float> resAngle = Matrix<float>(image.n_rows, image.n_cols);
// vector<Matrix<float>> resVect  = vector<Matrix<float>>(2);
    // Image kernel(3,3);
    // int chnl = 0;

    float X = 0, Y = 0;
    float module = 0, angle = 0;
    float xxx = 0;
    int iii =0;

    std::vector<float> kernelX = {1.f, 0.f, -1.f,/*вторая строка*/ 2.f, 0.f, -2.f,/*третья строка*/ 1.f, 0.f, -1.f} ;
        std::vector<float> kernelY = {1.f, 2.f, 1.f,/*вторая строка*/ 0.f, 0.f, 0.f,/*третья строка*/ -1.f, -2.f, -1.f} ;
        // std::vector<float> kernelY = {1, 2, 1,/*вторая строка*/ 0, 0, 0,/*третья строка*/ -1, -2, -1} ;
    for (uint i = 1; i < image.n_rows - 1 ; ++i)
    {
        for (uint j= 1; j < image.n_cols - 1; ++j)
        {

            X = 0, Y = 0, module  = 0, angle = 0;
            xxx = 0;

               for (uint k = 0; k < 3; ++k)
            {
                for (uint l = 0; l < 3; ++l)
                {
                    X += image(i - 1 + k, j - 1 + l) * kernelX.at(3 * k + l);

                }
            }             for (uint k = 0; k < 3; ++k)
            {
                for (uint l = 0; l < 3; ++l)
                {
                    Y += image(i - 1 + k, j - 1 + l) * kernelY.at(3 * k + l);

                }
            }
           
            module = hypot(X, Y);
            // module = sqrt(X*X + Y*Y);
            // module = X;
            if (X >  0.0) xxx = Y / X;
            if (X <  0.0) xxx = Y / X;
            if (xxx == 0) iii++;
            // angle = atan((xxx)) * 180.f / 3.14159265f  ;
            angle = atan2(Y, X) * 180.f / 3.14159265f;
            // cout << angle << "    ";
            if(module > 255.f) {module = 255.f;}
            if(module < 0 ){module = 0;}
            // if(angle > 255) {angle = 255;}
            // cout << module  << "    ";
            resModule(i,j) =module;
            // cout << module << "   ";
            resAngle(i,j) = angle;
        }
    }
        cout << "ВСЕГОГ НУЛЕЙ  " << iii << "  ";
// cout << "\n\n  " << resModule.n_rows;

// resVect.push_back(resModule);
// resVect.push_back(resAngle);
// cout << "\n Итоговый размер " <<resVect.at(0).n_rows;
return resModule;
}


//--------------------------------------------------------------------------------------
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
            /*p->Red = pix;
            p->Green= pix;
            p->Blue = pix;*/
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
 