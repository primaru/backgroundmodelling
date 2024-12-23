

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
wchar_t* projectPath;


Mat_<uchar> color2Gray(Mat_<Vec3b> src)
{
	int height = src.rows;
	int width = src.cols;
	Mat_<uchar> dst(height, width);
	dst.setTo(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i, j);
			uchar b = v3[0];
			uchar g = v3[1];
			uchar r = v3[2];
			dst.at<uchar>(i, j) = r * 0.2989 + g * 0.5870 + 0.1140 * b;
		}
	}
	return dst;
}
double meanValue(Mat_<uchar> src) {
	double avg = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			avg += src(i, j);
		}
	}
	avg = avg / double(src.rows * src.cols);
	return avg;
}
vector<int> calcHist(Mat_<uchar> src) {

	int height = src.rows;
	int width = src.cols;
	vector<int> h(256, 0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			h[src(i, j)] += 1;

		}
	}
	
	return h;

}
bool isInside(Mat img, int i, int j) {
	int height = img.rows;
	int width = img.cols;
	if (i < height && j < width) {
		if (i >= 0 && j >= 0)
			return true;
	}
	return false;

}
Mat_<uchar> erosion(Mat_<uchar> src) {
	Mat_<uchar>strel(2, 2);
	strel.setTo(0);
	Mat_<uchar> dst(src.rows, src.cols);
	dst.setTo(255);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 0) {
				for (int u = 0; u < strel.rows; u++) {
					for (int v = 0; v < strel.cols; v++) {
						if (strel(u, v) == 0) {
							int i2 = i + u - strel.rows / 2;
							int j2 = j + v - strel.cols / 2;
							if (isInside(dst, i2, j2)) {
								
								dst(i2, j2) = 0;
								
							}
						}
					}
				}
			}
		}
	}
	return dst;
}
vector<int> sumHist(vector<int> h1, vector<int> h2) {
	vector<int> h(256, 0);
	for (int i = 0; i < 256; i++) {
		h[i] = h1[i] + h2[i];
	}
	return h;
}
double updateLearningRate(double alpha, vector<int> h) {
	double sum = 0;
	double negal = 0;
	negal = 1 - alpha;

	for (int i = 0; i < 256; i++) {
		sum += h[i];
	}
	sum = (negal / sum + alpha) / 2;

	return sum;
}

Mat_<Vec3f> openImages2(string folder_path, double alpha, int T) {

	vector<cv::String> image_files;

	glob(folder_path, image_files);

	int cnt = 0;
	
	Vec3b threshold = { 0,0,0 };
	

	Mat_<Vec3f> testimg = imread(image_files[0]);
	Mat_<Vec3f> background(testimg.rows, testimg.cols);
	Mat_<Vec3f> dst(testimg.rows, testimg.cols);
	Mat_<Vec3f> diff(testimg.rows, testimg.cols);
	Mat_<Vec3f> diff2(testimg.rows, testimg.cols);
	Mat_<Vec3f> foreground(testimg.rows, testimg.cols);
	Mat_<uchar> graybck(testimg.rows, testimg.cols);
	Mat_<uchar> grayfore(testimg.rows, testimg.cols);
	Mat_<Vec3f> medianMatrix(testimg.rows, testimg.cols);
	Mat_<uchar>intFore(foreground.rows,foreground.cols);
	Vec3f black = { 0,0,0 };
	Vec3f white = { 255,255,255 };

	vector<int> h(256, 0), h2(256, 0), sumhist(256, 0);
	Vec3f median = { 0,0,0 };
	Mat frg;
	for (const auto& file : image_files) {

		Mat_<Vec3f> image = imread(file);
		Mat imt;
		image.convertTo(imt, CV_8UC3);
		imshow("Image", imt);

		dst = image;

		foreground.setTo(0);
		Mat_<Vec3f> n_background(testimg.rows, testimg.cols);
		n_background.setTo(0);
		diff.setTo(0);
		
		if (cnt == 0) {
			background.setTo(0);
		}
		else if (cnt < T) {
			for (int i = 0; i < background.rows; i++) {
				for (int j = 0; j < background.cols; j++) {
					background(i, j)[0] = background(i, j)[0] + dst(i, j)[0] / T;
					background(i, j)[1] = background(i, j)[1] + dst(i, j)[1] / T;
					background(i, j)[2] = background(i, j)[2] + dst(i, j)[2] / T;
				}
			}
		}
		else if (cnt >= T) {
			graybck = color2Gray(background);
			double meanv = meanValue(graybck);
			for (int i = 0; i < background.rows; i++) {
				for (int j = 0; j < background.cols; j++) {

					diff(i, j)[0] = abs(dst(i, j)[0] - background(i, j)[0]);
					diff(i, j)[1] = abs(dst(i, j)[1] - background(i, j)[1]);
					diff(i, j)[2] = abs(dst(i, j)[2] - background(i, j)[2]);
				
					threshold[0] = (dst(i, j)[0] + meanv) / 4;
					threshold[1] = (dst(i, j)[1] + meanv) / 4;
					threshold[2] = (dst(i, j)[2] + meanv) / 4;
					if (diff(i, j)[0] > threshold[0] && diff(i, j)[1] > threshold[1] || (diff(i, j)[0] > threshold[0] && diff(i, j)[2] > threshold[2]) || (diff(i, j)[1] > threshold[1] && diff(i, j)[2] > threshold[2])) {
						foreground(i, j) = white;
					}
				}
			}
			cvtColor(foreground, frg, COLOR_BGR2GRAY);
			frg = erosion(frg);
			graybck = color2Gray(background);
			h = calcHist(frg);
			h2 = calcHist(graybck);
			sumhist = sumHist(h, h2);

			alpha = updateLearningRate(alpha, sumhist);
			for (int i = 0; i < background.rows; i++) {
				for (int j = 0; j < background.cols; j++) {
					if (frg.at<uchar>(i,j) == 0) {
						background(i, j)[0] = alpha * dst(i, j)[0] + (1 - alpha) * background(i, j)[0];
						background(i, j)[1] = alpha * dst(i, j)[1] + (1 - alpha) * background(i, j)[1];
						background(i, j)[2] = alpha * dst(i, j)[2] + (1 - alpha) * background(i, j)[2];
					}
				}
			}
		}

		cnt++;
		// COMMENT THESE WHEN RUNNING FOR TESTS
		Mat srcb;
		background.convertTo(srcb, CV_8UC3);
		if(cnt>T)
			imshow("fore", frg);
		imshow("bgd", srcb);

		if(cnt > 330 && cnt < 400)
		waitKey();

	}
	Mat rez;
	background.convertTo(rez, CV_8UC3);
	return rez;
}

float testProiect(Mat_<Vec3f> src, Mat_<Vec3f> truth) {
	float result = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if ((abs(src(i, j)[0] - truth(i, j)[0])+abs(src(i, j)[1] - truth(i, j)[1])+abs(src(i, j)[2] - truth(i, j)[2]))/3 < 10) {
				result++;
			}
		}
	}
	result = result / (float)(src.rows * src.cols);
	return result;
}
float helper(string folder_path, string folder_path2, double alpha, int T) {
	vector<cv::String> image_files;
	float accuracy = 0;
	glob(folder_path, image_files);
	for (const auto& file : image_files) {
		Mat_<Vec3f> truth = imread(file);
		Mat_<Vec3f> src = openImages2(folder_path2,alpha,T);
		accuracy = testProiect(src, truth);
		Mat rez,rez2;
		truth.convertTo(rez2, CV_8UC3);
		src.convertTo(rez, CV_8UC3);
		imshow("truth", rez2);
		imshow("got", rez);
		waitKey(0);
		//cout << "result= " << accuracy;
	}
	waitKey(0);
	return accuracy;

}
int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);
	string folder_path = "A:/basic/512/*.jpg";
	string folder_path2 = "A:/basic/512/input/*.jpg";
	double alpha = 0.01;
	int T = 70;
	helper(folder_path,folder_path2,0.01,231);

	// UNCOMMENT THESE FOR TESTS

	/*cout << "for blurred \n";
	cout<<"for learning rate=0.01"<<" T=74 (70%)"<<" "<<helper(folder_path, folder_path2, 0.01, 74) << "\n";
	cout<< "for learning rate= 0.01" << " T=53 (50%)" << " " << helper(folder_path, folder_path2, 0.01, 53)<<"\n";
	cout << "for learning rate=0.01" << " T=31 (30%)" << " " << helper(folder_path, folder_path2, 0.01, 31) << "\n";
	cout << "for learning rate= 0.05" << " T=74 (70%)" << " " << helper(folder_path, folder_path2, 0.05, 74) << "\n";
	cout << "for learning rate= 0.5" << " T=53 (50%)" << " " << helper(folder_path, folder_path2, 0.5, 53) << "\n";
	cout<< "for learning rate= 0.75" << " T=31 (30%)" << " " << helper(folder_path, folder_path2, 0.75, 31)<<"\n";
	cout << endl;*/
	
	/*cout << "for 512 \n";
	folder_path = "A:/basic/512/*.jpg";
	folder_path2 = "A:/basic/512/input/*.jpg";

	cout << "for learning rate=0.01" << " T=323 (70%)" << " " << helper(folder_path, folder_path2, 0.01, 323) << "\n";
	cout << "for learning rate= 0.01" << " T=231 (50%)" << " " << helper(folder_path, folder_path2, 0.01, 231) << "\n";
	cout << "for learning rate= 0.01" << " T=137 (30%)" << " " << helper(folder_path, folder_path2, 0.01, 137) << "\n";
	cout << "for learning rate= 0.5" << " T=323 (70%)" << " " << helper(folder_path, folder_path2, 0.5, 323) << "\n";
	cout << "for learning rate= 0.25" << " T=231 (50%)" << " " << helper(folder_path, folder_path2, 0.25, 231) << "\n";
	cout << "for learning rate= 0.005" << " T=137 (30%)" << " " << helper(folder_path, folder_path2, 0.005, 137) << "\n";

	cout << endl;*/ 
	/*folder_path = "A:/basic/busStation/*.jpg";
	folder_path2 = "A:/basic/busStation/input/*.jpg";
	cout << "for busSation \n";
	cout << "for learning rate=0.01" << " T=431 (70%)" << " " << helper(folder_path, folder_path2, 0.01, 431) << "\n";
	cout << "for learning rate= 0.01" << " T=308 (50%)" << " " << helper(folder_path, folder_path2, 0.01, 308) << "\n";
	cout << "for learning rate= 0.01" << " T=185 (30%)" << " " << helper(folder_path, folder_path2, 0.01, 185) << "\n";
	cout << "for learning rate= 1" << " T=431 (70%)" << " " << helper(folder_path, folder_path2, 1, 431) << "\n";
	cout << "for learning rate= 0.5" << " T=308 (50%)" << " " << helper(folder_path, folder_path2, 0.5, 308) << "\n";
	cout << "for learning rate= 0.1" << " T=185 (30%)" << " " << helper(folder_path, folder_path2, 0.1, 185) << "\n";*/

	//cout << "for CameraParameter \n";
	//folder_path = "A:/basic/CameraParameter/GT_background1.jpg";
	//folder_path2 = "A:/basic/CameraParameter/input/*.jpg";

	//cout << "for learning rate=0.01" << " T=323 (70%)" << " " << helper(folder_path, folder_path2, 0.01, 323) << "\n";
	//cout << "for learning rate= 0.01" << " T=231 (50%)" << " " << helper(folder_path, folder_path2, 0.01, 231) << "\n";
	//cout << "for learning rate= 0.01" << " T=137 (30%)" << " " << helper(folder_path, folder_path2, 0.01, 137) << "\n";
	//cout << "for learning rate= 0.5" << " T=323 (70%)" << " " << helper(folder_path, folder_path2, 0.5, 323) << "\n";
	//cout << "for learning rate= 0.5" << " T=231 (50%)" << " " << helper(folder_path, folder_path2, 0.5, 231) << "\n";
	//cout << "for learning rate= 0.5" << " T=137 (30%)" << " " << helper(folder_path, folder_path2, 0.5, 137) << "\n";*/


	waitKey(0);


	//openImages2();
	return 0;
}