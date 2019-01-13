#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // cv::imread()
#include <opencv2/imgproc/imgproc.hpp> // cv::Canny
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/nonfree/nonfree.hpp> // add the link opencv_nonfree246d.lib
#include <opencv2/ml/ml.hpp> // Normal Bayes Classifier and Support Vector Machines (opencv_ml246d.lib)

#include <sstream>
#include <fstream>
#include <iostream> 

#include <atlstr.h>
//#include <math.h>
#include <string>

#define TEST_FILE_LIST "data\\cars_test.txt"
#define TRAIN_FILE_LIST "data\\cars_train.txt"
#define SEARCH_EXPRESSION "data\\*.png"

#define FEATURE_DETECTOR "SIFT" //Choise the feature detector "SIFT" or "SURF"

#define DEBUG_TEX false // if true print the debug text
#define DEBUG_IMG false // if true display the imagens
#define DEBUG_MAT false // if true save the matrix to file

#define DICTIONARY_MATRIX_FILE true // if false save the dictionery matrix to file
#define TRAIN_MATRIX_FILE true // if false save the train matrix to file
#define IMG_FILE false // if false save the result image merge with the mask to file
#define WIND_FILE false // if false save the detected windows locations
#define EVOLUATION_PROPORTION 50.0 // the percentege of fragment image that contains the car

#define TXT 0 // type of file
#define YML 1

using namespace cv;
using namespace std;

bool evalutateWindow(int x, int y, int size, Mat imagemask, double eval);
//Join the masks where green and black becomes black and the red becomes white
bool imgMaskMerge(map<int,vector<string>> &filesMap, vector<int> &fileList, map<int,Mat> &masksMap);
bool openImage(const string &filename, Mat &image);
bool readMatFromFile(Mat& m, string filename, int typeFile);
bool writeMatToFile(Mat& m, string filename, int typeFile);
int readFileListFromFile(vector<int> &fileList, const char* filename);
bool detectMaskFiles(vector<string> &filelist, map<int,vector<string>> &filesMap);
bool detectDirFiles(vector<string> &filelist);
