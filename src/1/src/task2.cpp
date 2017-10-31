#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

#include "Usable.h"
#include "timer.h"

#ifdef DEBUG
#include <glog/logging.h>
#include <iomanip>

#endif

/**
 * @mainpage Progect2: Описание
 *
 * На основе задания машграфа опробированы векторное расширение процессора и система документации Doxygen.

SSE использван в применении филтров Собеля (оператор свертки):
    task2.cpp -> std::vector<float> calculateHog(BMP &img)
и вычислении нормы градиента:
    usual.cpp -> double ConvolutionOp::operator()(const Matrix<double> &neighbourhood) const
Производительность была замерена на надоре картинок из задания по машграфу. На этих 345 файлах суммароное время вычисления дескриптора составило (среднее по 3 запускам):
    - 0.3636 -- SSE [0.001053 per image]
    - 0.3490 -- original [0.001011 per image]
Отсутствие выигрыша обосновывается дополнительными временными затрарами на загрузку данных в m128d переменные и выгрузку обратно в double. А 2 пременных за одну интсрукцию -- не столь большой выигрыш при таких затратах. В качестве решения предлагается переписать бальшую часть кода с использованием sse, что затронит логику вычислений, да и выходит за необходимые условия Задания.

С использованием Doxygen составлена документация к коду.
Note: Как коворится, на вопрос "что" должен отвечать сам код, а документация должна отвечать на вопрос "почему". Но "почему" расписано на 14 страницах задания по машграфу и нету смысла заниматься копипастом.
 */

/**
 * @file
 * @author Mikhail Agranovskiy, 321, cs msu
 */

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

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

//**********************************Okay, my code starts here********************************************

/// source image will be splited to thet number of squares
constexpr uint8_t N_SQUARES_PER_LINE = 8;
/// size of histogram (number of sections in 2pi interval)
constexpr uint8_t HIST_SZ = 8;

/// assume same-sized matrixes as params
std::vector<double> calcHistogramHog(const Matrix<double> &square,
                                     const Matrix<double> &abs,
                                     const Matrix<double> &angles)
{
    std::vector<double> hist(HIST_SZ, static_cast<double>(0));
    for (uint i = 0; i < square.n_rows; i++) {
        for (uint j = 0; j < square.n_cols; j++) {
            double tmpIdx = (static_cast<double>(M_PI) + angles(i, j)) * HIST_SZ / 2 / M_PI;
            uint idx = uint(tmpIdx) % HIST_SZ;
            hist[idx] += abs(i, j);
        }
    }
    return hist;
}

/// normalize histogram of dubles
void normaliseHist(vector<double> &hist)
{
    double norm = 0;
    for (const auto &elem : hist) {
        norm += elem * elem;
    }
    if (norm > std::numeric_limits<double>::epsilon()) {
        norm = std::sqrt(norm);
        for (auto &elem : hist) {
            elem /= norm;
        }
    }
}
/**
 * Calculate HOG descriptor for source image.
 *
 * SSE intrinsics are used in implementation.
 * @param img source image
 * @return HOG descriptor
 */
std::vector<float> calculateHog(BMP &img)
{
    auto n = static_cast<uint>(img.TellHeight());
    auto m = static_cast<uint>(img.TellWidth());
    n = n + (n % N_SQUARES_PER_LINE ? N_SQUARES_PER_LINE - n % N_SQUARES_PER_LINE : 0);
    m = m + (m % N_SQUARES_PER_LINE ? N_SQUARES_PER_LINE - m % N_SQUARES_PER_LINE : 0);

    // part1
    auto imgMatrix = extraMatrix(grayscale(img), n, m);

    // part2: Sobel convolution
    auto xProj = sobel_x(imgMatrix, false);
    auto yProj = sobel_y(imgMatrix, false);

    // part3: calculate gradients
    /// gradients absolute values
    Matrix<double> abs(n, m);
    /// gradients directions
    Matrix<double> angles(n, m);
    __m128d absSse;
    double absUsual[2];
    for (uint i = 0; i < n; i++) {
        for (uint j = 0; j < m; j++) {
            absSse = _mm_setr_pd(xProj(i, j), yProj(i, j));
            absSse = _mm_mul_pd(absSse, absSse);
            _mm_storeu_pd(absUsual, absSse);
            abs(i, j) = absUsual[0] + absUsual[1];

            // abs(i, j) = std::sqrt(std::pow(xProj(i, j), 2) + std::pow(yProj(i, j), 2));

            angles(i, j) = std::atan2(yProj(i,j), xProj(i, j));
        }
    }

    // part4: calculate histograms
    assert(n >= N_SQUARES_PER_LINE);
    assert(m >= N_SQUARES_PER_LINE);
    // iterate over squares
    std::vector<float> desc;
    for (uint i = 0, iStep = n / N_SQUARES_PER_LINE; i + iStep <= n; i += iStep) {
        for (uint j = 0, jStep = m / N_SQUARES_PER_LINE; j + jStep <= m; j += jStep) {
            auto hist = calcHistogramHog(imgMatrix.submatrix(i, j, iStep, jStep),
                                         abs.submatrix(i, j, iStep, jStep),
                                         angles.submatrix(i, j, iStep, jStep));
            // part5: normalise hists
            normaliseHist(hist);
            // part6: concatenate
            desc.insert(desc.end(), hist.begin(), hist.end());
        }
    }
    return desc;
}

/**
 * Extract features from dataset.
 * @param data_set vector of pairs <image, lable>
 * @param features vector of gistograms and lables for images from data_set.
 *                  The main aim of the function is to construct that vector.
 */
void ExtractFeatures(const TDataSet& data_set, TFeatures* features)
{
    double timeSum = 0;
    for (const auto &elem : data_set) {
        auto &img = *(elem.first);  // reference is not const because of BMP class architecture
        const auto &label = elem.second;

        // check input
        assert(img.TellHeight() <= static_cast<long long int>(std::numeric_limits<uint>::max()) && img.TellHeight() >= 0);
        assert(img.TellWidth() <= static_cast<long long int>(std::numeric_limits<uint>::max()) && img.TellWidth() >= 0);

        Timer t;

        t.start();
        auto hogDesc = calculateHog(img);
        timeSum += t.ret();
        features->emplace_back(std::make_pair(hogDesc, label));
    }
    std::cout << "Summary time: " << timeSum << std::endl;

}

//**********************************End of my code********************************************


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

//int main(int argc, char** argv) {
//#ifdef DEBUG
//    google::InitGoogleLogging(argv[0]);
//    google::InstallFailureSignalHandler();
//#endif
//
//    // Command line options parser
//    ArgvParser cmd;
//        // Description of program
//    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
//        // Add help option
//    cmd.setHelpOption("h", "help", "Print this help message");
//        // Add other options
//    cmd.defineOption("data_set", "File with dataset",
//        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
//    cmd.defineOption("model", "Path to file to save or load model",
//        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
//    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
//        ArgvParser::OptionRequiresValue);
//    cmd.defineOption("train", "Train classifier");
//    cmd.defineOption("predict", "Predict dataset");
//
//        // Add options aliases
//    cmd.defineOptionAlternative("data_set", "d");
//    cmd.defineOptionAlternative("model", "m");
//    cmd.defineOptionAlternative("predicted_labels", "l");
//    cmd.defineOptionAlternative("train", "t");
//    cmd.defineOptionAlternative("predict", "p");
//
//        // Parse options
//    int result = cmd.parse(argc, argv);
//
//        // Check for errors or help option
//    if (result) {
//        cout << cmd.parseErrorDescription(result) << endl;
//        return result;
//    }
//
//        // Get values
//    string data_file = cmd.optionValue("data_set");
//    string model_file = cmd.optionValue("model");
//    bool train = cmd.foundOption("train");
//    bool predict = cmd.foundOption("predict");
//
//        // If we need to train classifier
//    if (train)
//        TrainClassifier(data_file, model_file);
//        // If we need to predict data
//    if (predict) {
//            // You must declare file to save images
//        if (!cmd.foundOption("predicted_labels")) {
//            cerr << "Error! Option --predicted_labels not found!" << endl;
//            return 1;
//        }
//            // File to save predictions
//        string prediction_file = cmd.optionValue("predicted_labels");
//            // Predict data
//        PredictData(data_file, model_file, prediction_file);
//    }
//}