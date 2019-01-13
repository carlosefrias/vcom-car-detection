#include "Car_Detection.h"

int main(int argc, char** argv)
{
	initModule_nonfree();

	vector<string> filelist;
	map<int,vector<string>> filesMap;
	vector<int> fileTrainList;
	vector<int> fileTestList;
	map<int,Mat> masksMap;
	string featureDetector_str(FEATURE_DETECTOR);

	if (detectDirFiles(filelist))
	{
		if(DEBUG_TEX) cout << "Detected " << filelist.size() << " files."<< endl;
		detectMaskFiles(filelist, filesMap);
		if(DEBUG_TEX) cout << "Detected " << filesMap.size() << " images."<< endl;
	}
	else
		return -1;

	readFileListFromFile(fileTrainList, TRAIN_FILE_LIST);
	if(DEBUG_TEX) cout << "Listed " << fileTrainList.size() << " files for training."<< endl;
	readFileListFromFile(fileTestList, TEST_FILE_LIST);
	if(DEBUG_TEX) cout << "Listed " << fileTestList.size() << " files for testing."<< endl;

	//create Sift feature point extracter
	Ptr<FeatureDetector> detector = FeatureDetector::create(FEATURE_DETECTOR);
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(FEATURE_DETECTOR);  
    //create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(100, TermCriteria(), 1, KMEANS_PP_CENTERS);
	//create BoW (or BoF) descriptor extractor
    BOWImgDescriptorExtractor bowExtractor(detector, matcher);

	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	//To store the cluster feature vectors
	Mat dictionary;

	//Create the ditionary file
	if(!DICTIONARY_MATRIX_FILE)
	{
		for (unsigned int i=0; i<filesMap.size(); i++)
		{
			Mat imgTrainSet;
			string imgTrainName = filesMap[i+1].at(0);
			openImage("data\\"+imgTrainName, imgTrainSet);
			cout << "Processing image " << i+1 << "/" << filesMap.size() << ": " << imgTrainName << endl;
			if(DEBUG_IMG)
			{
				imshow("Train Image",imgTrainSet);
				waitKey(1);
				//destroyWindow("Train Image");
			}
			if(DEBUG_TEX) cout << "Detect " << FEATURE_DETECTOR << " keypoints... " << endl;
			//To store the keypoints that will be extracted by SIFT
			vector<KeyPoint> keypointsSet;
			//Detect SIFT keypoints (or feature points)
			detector->detect(imgTrainSet, keypointsSet);
			if(DEBUG_TEX) cout << "Extract descriptor... " << endl;
			//To store the SIFT descriptor of current image
			Mat DescriptorSet;        
			//extract descriptor from given image
			extractor->compute(imgTrainSet,keypointsSet,DescriptorSet);

			//put the all feature descriptors in a single Mat object
			featuresUnclustered.push_back(DescriptorSet);
		}
		if(DEBUG_TEX) cout << "All images featuresUnclustered size:" << featuresUnclustered.size() << endl;

		//cluster the feature vectors
		cout << "Creating and writing dicionary to file. " << endl;
		dictionary = bowTrainer.cluster(featuresUnclustered);
		//store the vocabulary
		writeMatToFile(dictionary, "Matrix_"+featureDetector_str+"_dictionary",YML);
	}
	else if (DICTIONARY_MATRIX_FILE) //Using Saved dicionary file
	{
		cout << "Reading dicionary from file. " << endl;
		//prepare BOW descriptor extractor from the dictionary
		readMatFromFile(dictionary, "Matrix_"+featureDetector_str+"_dictionary",YML);
	}
	destroyAllWindows();

	cout << "Setting dictionary ..." << endl;
    //Set the dictionary with the vocabulary we created in the first step
    bowExtractor.setVocabulary(dictionary);
	
	// Store de data for the training of classifier
	Mat TrainingData, BowTrainingData, ResTrainingData, TrainingData_aux;
	if(!TRAIN_MATRIX_FILE)
	{
		for (unsigned int i=0; i<fileTrainList.size(); i++)
		{
			Mat imgTrainSet;
			string imgTrainName = filesMap[fileTrainList.at(i)].at(0);
			openImage("data\\"+imgTrainName, imgTrainSet);
			cout << "Processing image " << i+1 << "/" << fileTrainList.size() << ": " << imgTrainName << endl;

			//Join the masks where green and black becomes black and the red becomes white
			Mat imagemask, imagemask_not, bw_aux1, bw_aux2, bw_aux3;
			openImage("data\\"+filesMap[fileTrainList.at(i)].at(1), bw_aux1);
			inRange(bw_aux1, Scalar(75), Scalar(77),imagemask);
			bw_aux1 = imagemask.clone();
			for (unsigned int j=2; j<filesMap[fileTrainList.at(i)].size(); j++)
			{
				openImage("data\\"+filesMap[fileTrainList.at(i)].at(j), bw_aux2);
				inRange(bw_aux2, Scalar(75), Scalar(77),bw_aux3);
				bitwise_or(bw_aux1, bw_aux3, imagemask);
				bw_aux1=imagemask;
			}

			if(DEBUG_TEX) cout << "Detect " << FEATURE_DETECTOR << " keypoints... " << endl;
			//To store the keypoints that will be extracted by SIFT
			vector<KeyPoint> keypointsSet, KeyPointBackground, keyPointCar;        
			//Detect SIFT keypoints (or feature points)
			//detector->detect(imgTrainSet, keypointsSet);
			if(DEBUG_TEX) cout << "Divide keypoints into classes... " << endl;
			detector->detect(imgTrainSet, keyPointCar, imagemask);
			bitwise_not(imagemask, imagemask_not);
			detector->detect(imgTrainSet, KeyPointBackground, imagemask_not);
			if(DEBUG_IMG) 
			{
				imshow("Mask train", imagemask);
				waitKey(1);
				//destroyWindow("Mask train");
			}

			if(DEBUG_TEX) cout << "Extract Bow descriptor... " << endl;
			//To store the BoW (or BoF) representation of current image
			Mat BowDescriptorBackground, BowDescriptorBackground_aux, BowDescriptorCar, BowDescriptorCar_aux, m_aux;        
			//extract BoW (or BoF) descriptor from current image car
			bowExtractor.compute(imgTrainSet,keyPointCar,BowDescriptorCar);
			//bowExtractor.compute(imgTrainSet,keyPointCar,BowDescriptorCar_aux);
			//extract BoW (or BoF) descriptor from current image background
			bowExtractor.compute(imgTrainSet,KeyPointBackground,BowDescriptorBackground);
			//bowExtractor.compute(imgTrainSet,KeyPointBackground,BowDescriptorBackground_aux);
			//The resulting bag-of-words representation vector should be normalized
			//normalize(BowDescriptorBackground_aux, BowDescriptorBackground,NORM_L1);
			//normalize(BowDescriptorCar_aux, BowDescriptorCar,NORM_L1);

			float car = 0, background = 1;
			ResTrainingData.push_back(car);
			ResTrainingData.push_back(background);
			vconcat(BowDescriptorCar, BowDescriptorBackground, m_aux);
			BowTrainingData.push_back(m_aux);
		}
		hconcat(BowTrainingData, ResTrainingData, TrainingData);
		cout << "Writing training matrix to file... " << endl;
		//store the vocabulary
		writeMatToFile(TrainingData, "Matrix_"+featureDetector_str+"_traning",YML);
	}
	else if(TRAIN_MATRIX_FILE)
	{
		cout << "Reading training matrix from file. " << endl;
		//prepare BOW descriptor extractor from the dictionary    
		readMatFromFile(TrainingData, "Matrix_"+featureDetector_str+"_traning",YML);
	}
	destroyAllWindows();

	Mat trainData = TrainingData.colRange(0, TrainingData.cols-1);
	Mat trainResponses = TrainingData.col(TrainingData.cols-1);

	if(DEBUG_MAT)
	{
		writeMatToFile(trainData, "Matrix_"+featureDetector_str+"_trainData",TXT);
		writeMatToFile(trainResponses, "Matrix_"+featureDetector_str+"_trainResponses",TXT);
	}

	NormalBayesClassifier nbayes;
	//Train Normal Bayes classifier...
	cout << "Train Normal Bayes classifier... " << endl;
	nbayes.train(trainData, trainResponses);

	CvSVM svm_auto;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC; //C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
	params.kernel_type = CvSVM::LINEAR; //LINEAR, POLY, RBF, SIGMOID
 
	//params.C= ; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    //params.nu= ; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    //params.p= ; // for CV_SVM_EPS_SVR
	//params.degree= ; // for poly
    //params.gamma= ;  // for poly/rbf/sigmoid
    //params.coef0= ;  // for poly/sigmoid

	//Train SVM classifier...
	cout << "Train SVM classifier... " << endl;
	svm_auto.train_auto(trainData, trainResponses, Mat(), Mat(), params);
	params = svm_auto.get_params();
	
	if(DEBUG_TEX) cout << "SVM type: " << params.svm_type << ", kernel: " << params.kernel_type << ", C: " << params.C << ", degree: " << params.degree << ", gamma: " << params.gamma << ", coef0: " << params.coef0 << endl;

	//#################################################
	//TESTING:
	//the location and scale of the cars are not known a priori it is necessary to analyse a
	//range of image locations and scales. A scanning window procedure should be used in this case.
	//Each window should be represented by a bag-of-words vector and tested with the previously trained
	//model. The result will be the areas of the image containing a car, possibly defined by bounding boxes.
	//#################################################
	int n_windows = 0; // number of sizes for window
	Mat windowsMaxPointBayes, windowsMaxPointSVM, windowsPointBayes, windowsPointSVM;
	if(!WIND_FILE)
	{
		for (unsigned int i=0; i<fileTestList.size(); i++)
		{
			Mat imgTestSet;
			string imgTestName = filesMap[fileTestList.at(i)].at(0);
			openImage("data\\"+imgTestName, imgTestSet);
			cout << "Processing image " << i+1 << "/" << fileTestList.size() << ": " << imgTestName << endl;

			if(DEBUG_IMG)
			{
				imshow("Test Image",imgTestSet);
				waitKey(1);
				//destroyWindow("Test Image");
			}
			int imgTestWidth = imgTestSet.size().width;
			int imgTestHeight = imgTestSet.size().height;
			Mat detectMaskSVM=imgTestSet.clone();
			Mat detectMaskBayes=imgTestSet.clone();
			Mat detectMaskSVMMax=imgTestSet.clone();
			Mat detectMaskBayesMax=imgTestSet.clone();
			//Make a scanning window procedure with diferents dimentions
			for(int sideSizeProportion=1; sideSizeProportion<7; sideSizeProportion++)
			{
				if(i<1)
					n_windows++;
				if(sideSizeProportion>3)
					sideSizeProportion=7;
				int width = imgTestWidth/sideSizeProportion;
				int height = imgTestHeight/sideSizeProportion;
				if(sideSizeProportion==1)
				{
					width = (imgTestWidth*65/100);
					height = (imgTestHeight*65/100);
				}
				int sideSize = (imgTestHeight>imgTestWidth) ? width : height;
				int increse=sideSize/6; // swaping speed of the window
				int x=0, y=0, CROPPING_WIDTH=sideSize, CROPPING_HEIGHT=sideSize; // initial values for the analizes window

				detectMaskSVM.setTo(Scalar(0)); // make mask all back
				detectMaskBayes.setTo(Scalar(0)); // make mask all back
				detectMaskSVMMax.setTo(Scalar(0)); // make mask all back
				detectMaskBayesMax.setTo(Scalar(0)); // make mask all back
				unsigned int maxPointsSVM=0;
				int xMaxSVM=0;
				int yMaxSVM=0;
				int percentageSVMMax=0;
				unsigned int maxPointsBayes=0;
				int xMaxBayes=0;
				int yMaxBayes=0;
				int percentageBayesMax=0;

				while(x<=imgTestWidth-sideSize)
				{
					while(y<=imgTestHeight-sideSize)
					{
						// cropping a small part of image
						Mat smallFrame = imgTestSet(Rect(x, y, CROPPING_WIDTH, CROPPING_HEIGHT));
						if(DEBUG_IMG) 
						{
							imshow("Small Frame", smallFrame);
							waitKey(1);
							//destroyWindow("Small Frame");
						}
						Mat imgWindow, BowDescriptorWindow, BowDescriptorWindow_aux;
						vector<KeyPoint> KeyPointWindow;
						//Detecting the keypoints/feature of image
						detector->detect(smallFrame, KeyPointWindow);
						if(KeyPointWindow.size()>0)
						{
							//Extracting the descriptor for the crop sample
							bowExtractor.compute(smallFrame,KeyPointWindow,BowDescriptorWindow);
							//bowExtractor.compute(smallFrame,KeyPointWindow,BowDescriptorWindow_aux);
							//normalize(BowDescriptorWindow_aux, BowDescriptorWindow,NORM_L1);

							//Normal Bayes Predicts the response
							int resultBayes = (int) nbayes.predict(BowDescriptorWindow);
					
							//SVM Predicts the response
							int resultSVM = (int) svm_auto.predict(BowDescriptorWindow);

							if(DEBUG_TEX) cout << "KeyPoint size: " << KeyPointWindow.size() << " SVM Class type: " << resultSVM << " ; Bayes Class type: " << resultBayes << endl;
							//when the area belongs to the class car the same area on the mask will be painted with white
							if(resultSVM==0)
							{
								int percentage = (int) 100/sideSizeProportion;
								if(sideSizeProportion==1)
									percentage = 65;

								if(maxPointsSVM<KeyPointWindow.size())
								{
									maxPointsSVM=KeyPointWindow.size();
									xMaxSVM=x;
									yMaxSVM=y;
									percentageSVMMax=percentage;
								}
								Mat windowPointSVM = (Mat_<float>(1,6) << fileTestList.at(i), x, y, sideSize, KeyPointWindow.size(), percentage);
								windowsPointSVM.push_back(windowPointSVM);

								rectangle(detectMaskSVM, Rect(x, y, CROPPING_WIDTH, CROPPING_HEIGHT), Scalar(255), CV_FILLED);
							}
							if (resultBayes==0)
							{
								int percentage = (int) 100/sideSizeProportion;
								if(sideSizeProportion==1)
									percentage = 65;

								if(maxPointsBayes<KeyPointWindow.size())
								{
									maxPointsBayes=KeyPointWindow.size();
									xMaxBayes=x;
									yMaxBayes=y;
									percentageBayesMax=percentage;
								}
								Mat windowPointBayes = (Mat_<float>(1,6) << fileTestList.at(i), x, y, sideSize, KeyPointWindow.size(), percentage);
								windowsPointBayes.push_back(windowPointBayes);

								rectangle(detectMaskBayes, Rect(x, y, CROPPING_WIDTH, CROPPING_HEIGHT), Scalar(255), CV_FILLED);
							}
						}
						y+=increse;
					}
					y=0;
					x+=increse;
				}
				Mat windowMaxPointSVM = (Mat_<float>(1,6) << fileTestList.at(i), xMaxSVM, yMaxSVM, sideSize, maxPointsSVM, percentageSVMMax);
				Mat windowMaxPointBayes = (Mat_<float>(1,6) << fileTestList.at(i), xMaxBayes, yMaxBayes, sideSize, maxPointsBayes, percentageBayesMax);
				windowsMaxPointSVM.push_back(windowMaxPointSVM);
				windowsMaxPointBayes.push_back(windowMaxPointBayes);
				
				rectangle(detectMaskSVMMax, Rect(xMaxSVM, yMaxSVM, CROPPING_WIDTH, CROPPING_HEIGHT), Scalar(255), CV_FILLED);
				rectangle(detectMaskBayesMax, Rect(xMaxBayes, yMaxBayes, CROPPING_WIDTH, CROPPING_HEIGHT), Scalar(255), CV_FILLED);

				Mat detectAreaImgSVM = imgTestSet.clone();
				Mat detectAreaImgBayes = imgTestSet.clone();
				Mat detectAreaImgSVMMax = imgTestSet.clone();
				Mat detectAreaImgBayesMax = imgTestSet.clone();
				//Merging the image with the mask
				detectAreaImgSVM &= detectMaskSVM;
				detectAreaImgBayes &= detectMaskBayes;
				detectAreaImgSVMMax &= detectMaskSVMMax;
				detectAreaImgBayesMax &= detectMaskBayesMax;
				if(!IMG_FILE)
				{
					cout << "Save the result image to file..." << endl;
					int percentage = (int) 100/sideSizeProportion;
					if(sideSizeProportion==1)
						percentage = 65;
					string smallWindowSize = to_string(percentage);
					imwrite("output/carsgraz_"+to_string(fileTestList.at(i))+"_"+featureDetector_str+"_svm_"+smallWindowSize+".jpg", detectAreaImgSVM);
					imwrite("output/carsgraz_"+to_string(fileTestList.at(i))+"_"+featureDetector_str+"_bayes_"+smallWindowSize+".jpg", detectAreaImgBayes);
					imwrite("output/carsgraz_"+to_string(fileTestList.at(i))+"_"+featureDetector_str+"_svm_Max_"+smallWindowSize+".jpg", detectAreaImgSVMMax);
					imwrite("output/carsgraz_"+to_string(fileTestList.at(i))+"_"+featureDetector_str+"_bayes_Max_"+smallWindowSize+".jpg", detectAreaImgBayesMax);
				}
				if(DEBUG_IMG) 
				{
					imshow("Mask test SVM", detectAreaImgSVM);
					waitKey(1);
					//destroyWindow("Mask test SVM");
					imshow("Mask test Bayes", detectAreaImgBayes);
					waitKey(1);
					//destroyWindow("Mask test Bayes");
					imshow("Mask test SVM Max", detectAreaImgSVMMax);
					waitKey(1);
					//destroyWindow("Mask test SVM");
					imshow("Mask test Bayes Max", detectAreaImgBayesMax);
					waitKey(1);
					//destroyWindow("Mask test Bayes");
				}
			}
			destroyAllWindows();
		}
		cout << "Writing Results object to a binary file... " << endl;
		writeMatToFile(windowsMaxPointSVM, "Matrix_"+featureDetector_str+"_windowsMaxPointSVM",YML);
		writeMatToFile(windowsMaxPointBayes, "Matrix_"+featureDetector_str+"_windowsMaxPointBayes",YML);
		writeMatToFile(windowsPointSVM, "Matrix_"+featureDetector_str+"_windowsPointSVM",YML);
		writeMatToFile(windowsPointBayes, "Matrix_"+featureDetector_str+"_windowsPointBayes",YML);
	}
	else if(WIND_FILE)
	{
		cout << "Reading Results object from binary file... " << endl;
		readMatFromFile(windowsMaxPointSVM, "Matrix_"+featureDetector_str+"_windowsMaxPointSVM",YML);
		readMatFromFile(windowsMaxPointBayes, "Matrix_"+featureDetector_str+"_windowsMaxPointBayes",YML);
		readMatFromFile(windowsPointSVM, "Matrix_"+featureDetector_str+"_windowsPointSVM",YML);
		readMatFromFile(windowsPointBayes, "Matrix_"+featureDetector_str+"_windowsPointBayes",YML);
	}
	
	//#################################################
	//EVALUATION: 
	//Compare at least two different approaches, different classifiers SVM auto and Normal Bayes, and different
	//local descriptors SURF and SIFT. To evaluate the performance in an image, compare the location of the car
	//given by your system with the mask provided in the dataset. If more than a given percentage of the pixels of the mask are inside the
	//bounding box, consider it a correct identification (true positive); otherwise, consider it an incorrect
	//identification (false positive). Only the testing images is used in the evaluation.
	//#################################################
	if(imgMaskMerge(filesMap, fileTestList, masksMap))
	{
		cout << "Join the masks where green and black becomes black and the red becomes white." << endl;
	}
	else
		return -1;

	cout << "Evaluation true positive and the false positive... " << endl;
	Mat evaluation = (Mat_<float>(2,2) << 0, 0, 0, 0); //[svm_true,svm_false;bayes_true,bayes_false]
	Mat evaluationMax = (Mat_<float>(2,2) << 0, 0, 0, 0); //[svm_true,svm_false;bayes_true,bayes_false]

	for (int i=0; i<windowsMaxPointSVM.rows; i++)
	{
		int n_mask = (int) windowsMaxPointSVM.at<float>(i,0);
		int x1 = (int) windowsMaxPointSVM.at<float>(i,1);
		int y1 = (int) windowsMaxPointSVM.at<float>(i,2);
		int mask_sideSize = (int) windowsMaxPointSVM.at<float>(i,3);
		if (evalutateWindow(x1, y1, mask_sideSize, masksMap[n_mask], EVOLUATION_PROPORTION))
			evaluationMax.at<float>(0,0) = evaluationMax.at<float>(0,0) + 1;
		else
			evaluationMax.at<float>(0,1) = evaluationMax.at<float>(0,1) + 1;
	}

	for (int i=0; i<windowsMaxPointBayes.rows; i++)
	{
		int n_mask = (int) windowsMaxPointBayes.at<float>(i,0);
		int x1 = (int) windowsMaxPointBayes.at<float>(i,1);
		int y1 = (int) windowsMaxPointBayes.at<float>(i,2);
		int mask_sideSize = (int) windowsMaxPointBayes.at<float>(i,3);
		if (evalutateWindow(x1, y1, mask_sideSize, masksMap[n_mask], EVOLUATION_PROPORTION))
			evaluationMax.at<float>(1,0) = evaluationMax.at<float>(1,0) + 1;
		else
			evaluationMax.at<float>(1,1) = evaluationMax.at<float>(1,1) + 1;
	}

	for (int i=0; i<windowsPointSVM.rows; i++)
	{
		int n_mask = (int) windowsPointSVM.at<float>(i,0);
		int x1 = (int) windowsPointSVM.at<float>(i,1);
		int y1 = (int) windowsPointSVM.at<float>(i,2);
		int mask_sideSize = (int) windowsPointSVM.at<float>(i,3);
		if (evalutateWindow(x1, y1, mask_sideSize, masksMap[n_mask], EVOLUATION_PROPORTION))
			evaluation.at<float>(0,0) = evaluation.at<float>(0,0) + 1;
		else
			evaluation.at<float>(0,1) = evaluation.at<float>(0,1) + 1;
	}

	for (int i=0; i<windowsPointBayes.rows; i++)
	{
		int n_mask = (int) windowsPointBayes.at<float>(i,0);
		int x1 = (int) windowsPointBayes.at<float>(i,1);
		int y1 = (int) windowsPointBayes.at<float>(i,2);
		int mask_sideSize = (int) windowsPointBayes.at<float>(i,3);
		if (evalutateWindow(x1, y1, mask_sideSize, masksMap[n_mask], EVOLUATION_PROPORTION))
			evaluation.at<float>(1,0) = evaluation.at<float>(1,0) + 1;
		else
			evaluation.at<float>(1,1) = evaluation.at<float>(1,1) + 1;
	}

	writeMatToFile(evaluation, "Matrix_"+featureDetector_str+"_evaluation", TXT);
	cout << "Evaluation matrix for " << n_windows << " sizes of windows using SVM e Bayes classifiers... " << endl;
	cout << "E = "<< endl << " "  << evaluation << endl;

	writeMatToFile(evaluationMax, "Matrix_"+featureDetector_str+"_evaluation_Max", TXT);
	cout << "Evaluation Max matrix for " << n_windows << " sizes of windows using SVM e Bayes classifiers... " << endl;
	cout << "E_Max = "<< endl << " "  << evaluationMax << endl;

	cout << endl << "Prime Enter para terminar a App..." << endl;
	getchar();
	return 0;
}

bool evalutateWindow(int x, int y, int size, Mat imagemask, double eval)
{
	// detecta X% da janela contem o carro
	int pix_count = 0, pix_car = 0;
	for(int i = x; i < x + size; i++)
	{
		for(int j = y; j < y + size; j++)
		{
			pix_count++;
			int color = (int) imagemask.at<uchar>(j, i);
			if (color == 255)
				pix_car++;
		}
	}
	double percentage_window = 100.0 * (pix_car / (double) pix_count);

	// detecta X% do carro contem a janela
	int pix_mask_car = 0;
	for(int k = 0; k < imagemask.rows; k++)
	{
		for(int l = 0; l < imagemask.cols; l++)
		{
			int color = (int) imagemask.at<uchar>(k, l);
			if (color == 255)
				pix_mask_car++;
		}
	}
	double percentage_car = 100.0 * (pix_car / (double) pix_mask_car);

	return (percentage_window >= eval || percentage_car >= eval) ? true : false;
}

bool imgMaskMerge(map<int,vector<string>> &filesMap, vector<int> &fileList, map<int,Mat> &masksMap)
{
	for (unsigned int i=0; i<fileList.size(); i++)
	{
		Mat imagemask, bw_aux1, bw_aux2, bw_aux3;
		openImage("data\\"+filesMap[fileList.at(i)].at(1), bw_aux1);
		inRange(bw_aux1, Scalar(75), Scalar(77),imagemask);
		bw_aux1 = imagemask.clone();
		string imgTestName = filesMap[fileList.at(i)].at(0);
		//cout << "Evaluating image " << i+1 << "/" << fileList.size() << ": " << imgTestName << endl;
		for (unsigned int j=2; j<filesMap[fileList.at(i)].size(); j++)
		{
			openImage("data\\"+filesMap[fileList.at(i)].at(j), bw_aux2);
			inRange(bw_aux2, Scalar(75), Scalar(77),bw_aux3);
			bitwise_or(bw_aux1, bw_aux3, imagemask);
			bw_aux1=imagemask;
		}
		masksMap[fileList.at(i)]=imagemask;
	}
	return true;
} 

bool openImage(const string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
	if( !image.data ) {
		cout << " --(!) Error reading image " << filename << endl;
		return false;
	}
	return true;
}

bool readMatFromFile(Mat& m, string filename, int typeFile)
{
	if (typeFile == YML)
	{
		filename.append(".yml");
		FileStorage fs(filename.c_str(), FileStorage::READ);
		fs["vocabulary"] >> m;
		fs.release();
	}
	else if (typeFile == TXT)
	{
		filename.append(".txt");
		ifstream fin(filename.c_str());
		String line;
		if(!fin)
		{
			cout<<"File Not Opened"<<endl;
			return false;
		}
		//Mat m_aux;
		while(fin.good())
		{
			getline(fin, line);

			stringstream s_elems(line);
			string elem;
			Mat elems;
			while (getline(s_elems, elem, ';'))
			{
				elems.push_back(stof(elem));
			}
			elems=elems.t();
			m.push_back(elems);
		}
		fin.close();
	}
	else
		return false;
	return true;
}

bool writeMatToFile(Mat& m, string filename, int typeFile)
{
	if (typeFile == YML)
	{
		filename.append(".yml");
		FileStorage fs(filename.c_str(), FileStorage::WRITE);
		fs << "vocabulary" << m;
		fs.release();
	}
	else if (typeFile == TXT)
	{
		filename.append(".txt");
		ofstream fout(filename.c_str());
		if(!fout)
		{
			cout<<"File Not Opened"<<endl;
			return false;
		}

		for(int i=0; i<m.rows; i++)
		{
			for(int j=0; j<m.cols; j++)
			{
				if (j!=m.cols-1)
					fout<<m.at<float>(i,j)<<";";
				else
					fout<<m.at<float>(i,j);
			}
			if(i<m.rows-1)
				fout<<endl;
		}
		fout.close();
	}
	else
		return false;
	return true;
}

int readFileListFromFile(vector<int> &fileList, const char* filename)
{
	ifstream fin(filename);
	String line;
    if(!fin)
    {
        cout<<"File Not Opened"<<endl;
		return 0;
    }
	while(fin.good())
	{
		getline(fin, line);
		if (line.compare(0, 9, "carsgraz_") == 0)
		{
			string elem(line, line.size()-3, 3);
			int number;
			istringstream (elem) >> number;
			fileList.push_back(number);
		}
	}
	fin.close();
	return fileList.size();
}

bool detectMaskFiles(vector<string> &filelist, map<int,vector<string>> &filesMap)
{
	int n_img = 1, est = 0;
	unsigned int i;
	for (i=0; i<filelist.size(); i++)
	{
		string filename = filelist.at(i);
		string fileImg = to_string(n_img) + ".image.png";
		string fileMsk = to_string(n_img-1) + ".mask.";
		if (((filename.compare(9, fileMsk.size(), fileMsk) == 0)
			| (filename.compare(10, fileMsk.size(), fileMsk) == 0)
			| (filename.compare(11, fileMsk.size(), fileMsk) == 0)) && n_img > 1)
		{
			filesMap[n_img-1].push_back(filename);
			//cout << filename << " - " << filesMap[n_img-1].size() << endl;
			est++;
		}
		else if (filename.compare(filename.size()-fileImg.size(), fileImg.size(), fileImg) == 0)
		{
			vector<string> imgMask;
			imgMask.push_back(filename);
			filesMap[n_img]=imgMask;
			n_img++;
			//cout << filename << endl;
			est++;
		}
	}
	if(DEBUG_TEX) cout << "Data added to Map: " << est << endl;
	return true;
}

bool detectDirFiles(vector<string> &filelist)
{
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError=0;
	hFind = FindFirstFile(TEXT(SEARCH_EXPRESSION), &ffd);
	
	if (INVALID_HANDLE_VALUE == hFind) 
	{
		cout << "Unable to detect files.!!!" << endl;
		return false;
	}
	else
	{
		/// List all the files in the directory with some info about them.
		do
		{
			if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)){}
			else if (~(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				///convert from wide char to narrow char array
				char ch[260];
				char DefChar = ' ';
				WideCharToMultiByte(CP_ACP,0,ffd.cFileName,-1, ch,260,&DefChar, NULL);
    
				///A std:string  using the char* constructor.
				string ss(ch);
				filelist.push_back (ss);
			}
		}
		while (FindNextFile(hFind, &ffd) != 0);

		dwError = GetLastError();
		if (dwError != ERROR_NO_MORE_FILES)
		{
			cout << "Error no more files!!!" << endl;
			return false;
		}
	}
	FindClose(hFind);
	return true;
}
