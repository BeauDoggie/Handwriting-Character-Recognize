#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>



using namespace cv;
using namespace std;

void read_img_idx3_from_file(string path, Mat& img);
void read_lab_idx1_from_file(string path, Mat& label); 



int main()
{

	//step1: parse the image label and images from data file;
	//format: idx3: train data;
	Mat test_images;
	Mat train_images;
	Mat test_labels;
	Mat train_labels;


	//Parse images from file.
	string fileTrainImgPath = "../../train data/train-images.idx3-ubyte";
	string fileTrainLabPath = "../../train data/train-labels.idx1-ubyte";
	string fileTestImgPath  = "../../test data/t10k-images.idx3-ubyte";
	string fileTestLabPath  = "../../test data/t10k-labels.idx1-ubyte";

	read_img_idx3_from_file(fileTrainImgPath, train_images);
	read_lab_idx1_from_file(fileTrainLabPath, train_labels);

	read_img_idx3_from_file(fileTestImgPath, test_images);
	read_lab_idx1_from_file(fileTestLabPath, test_labels);

	/*************KNN Algorithm create *****************/
	int k = 9;
	Ptr<ml::KNearest> knn = ml::KNearest::create();

	knn->setDefaultK(k);
	knn->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
	knn->setIsClassifier(true);
	

	/***********Create the KNN training samples*********/

	Ptr<ml::TrainData> train_data = ml::TrainData::create(train_images, ml::ROW_SAMPLE, train_labels);
	bool ret  = knn->train(train_data);



	cout << "test image size: " << test_images.rows << endl;
	Mat result(Size(1,10),CV_32F);

	//knn->predict(test_images, result);

	knn->findNearest(test_images(Range(0, 10), Range(0, test_images.size().width)), k, result);


	//compare(result, test_images, bresult,0);

	for (int i = 0; i < 10; i++)
	{
		cout << "test_images: " << test_labels.at<float>(i, 0) << endl;
		cout << "result: " << result.at<float>(i,0) << endl;

	}







	//cout << "train image size: " << train_images.size() << endl;
	//cout << "train label size: " << train_labels.size() << endl;

	//imshow("image", train_images.at(60000));
	//cout << "label: " << (int)train_labels.at(60000) << endl;

	waitKey(0);






#ifdef C_READ
	Ptr<FILE> fp_img(fopen(fileImgPath.c_str(), "rb"), fclose);
	Ptr<FILE> fp_lab(fopen(fileLabPath.c_str(), "rb"), fclose);
	
	if (fp_lab.empty())
		cout << "failed to open the train label file!" << endl;
	if (fp_img.empty())
		cout << "failed to open the train image file!" << endl;

	uint magic_num;
	uint image_num;
	uint img_w;
	uint img_h;

	/**********************read the image file header***********************/
	fread(&magic_num, sizeof(uint), 1, fp_img);
	fread(&image_num, sizeof(uint), 1, fp_img);
	fread(&img_h, sizeof(uint), 1, fp_img);
	fread(&img_w, sizeof(uint), 1, fp_img);
	/**********************read the label file header***********************/
	fread(&magic_num, sizeof(uint), 1, fp_lab);
	fread(&image_num, sizeof(uint), 1, fp_lab);

	int ret;
	uchar label;
	Mat  image(28, 28, CV_8UC1);

	while (1)
	{
		ret = fread(image.data, sizeof(uchar), 28 * 28, fp_img);
	    ret = fread(&label, sizeof(uchar), 1, fp_lab);
		if (!ret)
			break;
		train_images.push_back(image);
		train_labels.push_back(label);

		imshow("image", image);
		cout << "label: " << (int)label << endl;
		waitKey(1);
	}

	cout << "train_image is OK!" << endl;

	cout << "train image size: " << train_labels.size() << endl;

	
	waitKey(0);
#endif




	system("pause");



	return 1;
}




void read_img_idx3_from_file(string path, Mat& img)
{

	uchar header[4];
	uint  image_num;
	uint  image_w;
	uint  image_h;

	Mat   image(28,28,CV_8UC1);

	fstream iofile(path, ios::in | ios::out | ios::binary);

	if (iofile.bad())
	{
		cout << "failed to open path file!" << endl;
		return;
	}

	//read img header.
	iofile.read((char*)header, 4);//read the magic number.

	iofile.read((char*)header, 4);//read the image number.
	image_num = (uint)(header[3] + header[2] * (1 << 8) + header[1] * (1 << 16) + header[0] * (1 << 24));

	iofile.read((char*)header, 4);//read the image width.
	image_w = (uint)(header[3] + header[2] * (1 << 8) + header[1] * (1 << 16) + header[0] * (1 << 24));

	iofile.read((char*)header, 4);//read the image height.
	image_h = (uint)(header[3] + header[2] * (1 << 8) + header[1] * (1 << 16) + header[0] * (1 << 24));

	//cout << "image_num: " << image_num << endl;
	//cout << "image_w:  " << image_w << endl;
	//cout << "image_h: " << image_h << endl;

	img.create(Size(image_w * image_h, image_num), CV_8UC1);

	uchar* ptr_data = img.data;
	
	while (!iofile.eof())
	{
		iofile.read((char*)ptr_data, image_w * image_h);
		ptr_data += image_w * image_h;
	}
	
	img.convertTo(img, CV_32F);

	//cout << "img size: " << img.rows << endl;

}


void read_lab_idx1_from_file(string path, Mat& label)
{
	uint  label_num;
	uchar img_label;
	uchar header[4];
	fstream iofile(path, ios::in | ios::out | ios::binary);
	if (iofile.bad())
	{
		cout << "failed to open the label file!" << endl;
		return;
	}

	//read label file header.
	iofile.read((char*)header, 4);//read the magic number.

	iofile.read((char*)header, 4);//read the item number.
	label_num = (uint)(header[3] + header[2] * (1 << 8) + header[1] * (1 << 16) + header[0] * (1 << 24));

	label.create(Size(1, label_num), CV_8UC1);

	uchar* ptr_data = label.data;

	while (!iofile.eof())
	{
		iofile.read((char*)ptr_data++, 1);
	}

	label.convertTo(label, CV_32F);

	cout << "label row: " << label.rows << endl;
}