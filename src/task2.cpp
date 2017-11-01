#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
#include "usr2.h"


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

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<Image*, int> > TImageDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

inline double calc_brightness_gray(uch b, uch g, uch r){
    double a  = r * 0.299 + g * 0.587 + b * 0.114;
    return a;
}

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}
/*
void LoadImagesGrayScale(const TFileList& file_list, TImageDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset

        Image res = grayscale_from_BMP(*image);
        data_set->push_back(make_pair(&res, file_list[img_idx].second));
    }
}
*/
// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}
void GrayScale(const TDataSet& data_set){
    // getpixel(a,b) - a - width;  b -  height
        int borderH = 0, borderW = 0;
        uch clr = 0;
        for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {
        // cout << "1 pixel " << data_set.size() << " image " << image_idx;
            borderH = get<0>(data_set.at(image_idx))->TellHeight();
            for (int i = 0; i < borderH - 1; ++i)
            {
                // cout << "\n  i " << i << " borderW " << borderW << " borderH " << borderH << "size: height: " <<"\n";
                borderW = get<0>(data_set.at(image_idx))->TellWidth();
                for (int j = 0; j < borderW - 1; ++j)
                    {
                        // cout << " j " << j << "; width " << get<0>(data_set.at(image_idx))->TellWidth();
                        RGBApixel pix = get<0>(data_set.at(image_idx))->GetPixel(j,i);
                        clr = static_cast<unsigned char>(calc_brightness_gray(pix.Blue, pix.Green, pix.Red));
                        RGBApixel pixNew = RGBApixel();
                        pixNew.Red = clr;
                        pixNew.Green = clr;
                        pixNew.Blue = clr;
                        get<0>(data_set.at(image_idx))->SetPixel(j,i, pixNew);
                        // RGBApixel pix2 = get<0>(data_set.at(image_idx))->GetPixel(j,i);
                        // cout << " red " << (uint)pix2.Red << "  green " << (uint)pix2.Green << " blue " << (uint)pix2.Blue << "\n";
                    }    
            }

            // cout << image_idx <<"  pixel " <<static_cast<int>(get<0>(data_set.at(image_idx))->TellWidth()) << "\n ";
        }
}

void SobelConvolutionOld(TDataSet& data_set){
    std::vector<int> sobel_x = {-1, 0, 1,/*вторая строка*/ -2, 0, 2,/*третья строка*/ -1, 0, 1} ;
    std::vector<int> sobel_y = {-1, -2, -1,/*вторая строка*/ 0, 0, 0,/*третья строка*/ 1, 2, 1} ;
// getpixel(a,b) - a - width;  b -  height
    TDataSet data_set_y;
    std::vector<int> pixel(4);
    int borderH = 0, borderW = 0;
    // uch clr = 0;
    RGBApixel pix_x = RGBApixel();
    RGBApixel pix_y = RGBApixel();
    for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {
        data_set.at(image_idx) = make_pair(get<0>(data_set.at(image_idx)),get<1>(data_set.at(image_idx)));

        cout << "1 pixel - NEW dataset" << data_set_y.size() << " image " << image_idx;
        borderH = get<0>(data_set.at(image_idx))->TellHeight();
        for (int i = 1; i < borderH - 1; ++i)
        {
            // cout << "\n  i " << i << " borderW " << borderW << " borderH " << borderH << "size: height: " <<"\n";
            borderW = get<0>(data_set.at(image_idx))->TellWidth();
            for (int j = 1; j < borderW - 1; ++j)
                {
                    for (uint l = 0; l < 3; ++l)
                    {
                        for (uint k = 0; k < 3; ++k)
                        {
                            // pix = get<0>(data_set.at(image_idx))->GetPixel(j,i);
                            pix_x.Red +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Red * sobel_x.at(k * 3 + l);
                            pix_x.Green +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Green * sobel_x.at(k * 3 + l);
                            pix_x.Blue +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Blue * sobel_x.at(k * 3 + l);
                            pix_y.Red +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Red * sobel_y.at(k * 3 + l);
                            pix_y.Green +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Green * sobel_y.at(k * 3 + l);
                            pix_y.Blue +=  get<0>(data_set.at(image_idx))->GetPixel(j - 1 + k,i - 1 + l).Blue * sobel_y.at(k * 3 + l);

                            // pixel.at(chnl) = 
                             // += get<0> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                            // pixel.at(chnl + 1) += get<1> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                            // pixel.at(chnl + 2) += get<2> (image(i -1 + k,j - 1 + l)) * kernel.at(k * 3 + l);
                                                      
                        }
                    }
                    get<0>(data_set.at(image_idx))->SetPixel(j,i, pix_x);
                    get<0>(data_set_y.at(image_idx))->SetPixel(j,i, pix_y);




                    // cout << " j " << j << "; width " << get<0>(data_set.at(image_idx))->TellWidth();
                    // RGBApixel pix2 = get<0>(data_set.at(image_idx))->GetPixel(j,i);
                    // cout << " red " << (uint)pix2.Red << "  green " << (uint)pix2.Green << " blue " << (uint)pix2.Blue << "\n";
                }    
        }

        // cout << image_idx <<"  pixel " <<static_cast<int>(get<0>(data_set.at(image_idx))->TellWidth()) << "\n ";
    }    

}

void SobelConvolution(TDataSet& data_set){
    // int borderH = 0, borderW = 0;
    // uch clr = 0;
    for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {

    }
    }
// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    // GrayScale(data_set);
    features = features;
    Matrix<float> tempp = grayscale_Matrix_from_BMP(*data_set.at(0).first);
    Matrix<float> tempp2 = tempp;
    tempp = Sobel_X_Img(tempp);
    tempp2 = Sobel_Y_Img(tempp2);
    tempp2 = SobelConvolutionImg(tempp, tempp2);
    save_Matrix( tempp2, "/home/vorch/CGTESTS/1.bmp");
    cout << "Я сделаль!\n";


    /*BMP* temp ;
    for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {
        // *data_set.at(image_idx).first = 
        temp =new BMP();
        temp = data_set.at(image_idx).first;
        Rescale(temp,'w',64);
    cout << " новые размеры ищоюражегия 3: " << temp.TellWidth( )<< " и высота: "<< temp.TellHeight();
        
        Rescale(*data_set.at(image_idx).first,'h',64);
    cout << " новые размеры ищоюражегия 3: " << get<0>(data_set.at(image_idx))->TellWidth( )<< " и высота: "<< get<0>(data_set.at(image_idx))->TellHeight();
        if(get<0>(data_set.at(image_idx))->TellHeight() < 64) cout << "ПИДАРАСЫ!!!! "<< get<0>(data_set.at(image_idx))->TellHeight();
        delete &temp;
    }
    RGBApixel pix2 = get<0>(data_set.at(1))->GetPixel(20,20);
                        cout << " red " << static_cast<int>(pix2.Red) << "  green " <<static_cast<int>( pix2.Green) << " blue " << static_cast<int>( pix2.Blue) << "\n";
    *//*const char ci = 0;
std::cout << typeid(ci).name() << '\n';
        int i = 0;
    for (size_t image_idx = 0; image_idx < data_set.size()-1; ++image_idx) {
        i++;
        //  static_cast<char>(static_cast<int>( 
        // static_cast<char>(static_cast<int>(get<0>(data_set.at(1))->GetPixel(1,1).Red - 100))
        cout << i <<"  pixel " <<static_cast<int>(get<0>(data_set.at(image_idx))->TellWidth()) << "\n ";
        // PLACE YOUR CODE HERE
        // Remove this sample code and place your feature extraction code here
        vector<float> one_image_features;
        one_image_features.push_back(1.0);
        features->push_back(make_pair(one_image_features, 1));
        // End of sample code

    }*/
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}




// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2017.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}