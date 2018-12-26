#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudalegacy.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <chrono>

int main()
{

	// ----------------------------------------------------------------------------------------------------------------------------
	// GPU preparation
	// ----------------------------------------------------------------------------------------------------------------------------
	auto start = std::chrono::high_resolution_clock::now(); // Tick();
	cv::cuda::setDevice(0); // Prepare GPU 0
	auto elapsed = std::chrono::high_resolution_clock::now() - start; // Tock();

	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(); // Elapsed time in microsecond
	std::cout << "GPU initialization: " << microseconds << " microseconds" << std::endl;


	// ----------------------------------------------------------------------------------------------------------------------------
	// Load images
	// ----------------------------------------------------------------------------------------------------------------------------

	// Declare path and names of images
	const std::string pathImagesFolder = "./";

	const std::string pathWithName120x120_1 = pathImagesFolder + "120x120_1.png";
	const std::string pathWithName120x120_2 = pathImagesFolder + "120x120_2.png";

	const std::string pathWithName200x200_1 = pathImagesFolder + "200x200_1.png";
	const std::string pathWithName200x200_2 = pathImagesFolder + "200x200_2.png";

	const std::string pathWithName350x350_1 = pathImagesFolder + "350x350_1.png";
	const std::string pathWithName350x350_2 = pathImagesFolder + "350x350_2.png";

	const std::string pathWithName1000x1000_1 = pathImagesFolder + "1000x1000_1.png";
	const std::string pathWithName1000x1000_2 = pathImagesFolder + "1000x1000_2.png";

	const std::string pathWithName12500x7800_1 = pathImagesFolder + "12500x7800_1.png";
	const std::string pathWithName12500x7800_2 = pathImagesFolder + "12500x7800_2.png";

	// Load images into OpenCV Mat object
	cv::Mat img120x120_1 = cv::imread(pathWithName120x120_1);
	cv::Mat img120x120_2 = cv::imread(pathWithName120x120_2);

	cv::Mat img200x200_1 = cv::imread(pathWithName200x200_1);
	cv::Mat img200x200_2 = cv::imread(pathWithName200x200_2);

	cv::Mat img350x350_1 = cv::imread(pathWithName350x350_1);
	cv::Mat img350x350_2 = cv::imread(pathWithName350x350_2);

	cv::Mat img1000x1000_1 = cv::imread(pathWithName1000x1000_1);
	cv::Mat img1000x1000_2 = cv::imread(pathWithName1000x1000_2);

	cv::Mat img12500x7800_1 = cv::imread(pathWithName12500x7800_1);
	cv::Mat img12500x7800_2 = cv::imread(pathWithName12500x7800_2);

	// Check if all are ok
	if (img120x120_1.empty()) {
		std::cerr << "Can't open image " << pathWithName120x120_1 << std::endl;
		system("pause");
		exit(-1);
	}
	if (img120x120_2.empty()) {
		std::cerr << "Can't open image " << pathWithName120x120_2 << std::endl;
		system("pause");
		exit(-1);
	}

	if (img200x200_1.empty()) {
		std::cerr << "Can't open image " << pathWithName200x200_1 << std::endl;
		system("pause");
		exit(-1);
	}
	if (img200x200_2.empty()) {
		std::cerr << "Can't open image " << pathWithName200x200_2 << std::endl;
		system("pause");
		exit(-1);
	}

	if (img350x350_1.empty()) {
		std::cerr << "Can't open image " << pathWithName350x350_1 << std::endl;
		system("pause");
		exit(-1);
	}
	if (img350x350_2.empty()) {
		std::cerr << "Can't open image " << pathWithName350x350_2 << std::endl;
		system("pause");
		exit(-1);
	}

	if (img1000x1000_1.empty()) {
		std::cerr << "Can't open image " << pathWithName1000x1000_1 << std::endl;
		system("pause");
		exit(-1);
	}
	if (img1000x1000_2.empty()) {
		std::cerr << "Can't open image " << pathWithName1000x1000_2 << std::endl;
		system("pause");
		exit(-1);
	}

	if (img12500x7800_1.empty()) {
		std::cerr << "Can't open image " << pathWithName12500x7800_1 << std::endl;
		system("pause");
		exit(-1);
	}
	if (img12500x7800_2.empty()) {
		std::cerr << "Can't open image " << pathWithName12500x7800_2 << std::endl;
		system("pause");
		exit(-1);
	}


	//--------------------------------------------------------------------------------------------------------------
	// load into GPU
	//--------------------------------------------------------------------------------------------------------------
	std::cout << "------------------------------------------" << std::endl << std::endl;

	// Declaration of all variables we are going to need
	cv::cuda::GpuMat GPUimg120x120_1, GPUimg120x120_2, GPUimgGray120x120, GPUimgTemp120x120;
	cv::cuda::GpuMat GPUimg200x200_1, GPUimg200x200_2, GPUimgGray200x200, GPUimgTemp200x200;
	cv::cuda::GpuMat GPUimg350x350_1, GPUimg350x350_2, GPUimgGray350x350, GPUimgTemp350x350;
	cv::cuda::GpuMat GPUimg1000x1000_1, GPUimg1000x1000_2, GPUimgGray1000x1000, GPUimgTemp1000x1000;
	cv::cuda::GpuMat GPUimg12500x7800_1, GPUimg12500x7800_2, GPUimgGray12500x7800, GPUimgTemp12500x7800;

	cv::cuda::GpuMat temp1, temp2, temp3, temp4, temp5;

	// Load all gpu variables with Mat images
	
	start = std::chrono::high_resolution_clock::now();
	GPUimg120x120_1.upload(img120x120_1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Load 120x120 into GPU: " << microseconds << " microseconds" << std::endl;
	GPUimg120x120_2.upload(img120x120_2);
	
	start = std::chrono::high_resolution_clock::now();
	GPUimg200x200_1.upload(img200x200_1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Load 200x200 into GPU: " << microseconds << " microseconds" << std::endl;
	GPUimg200x200_2.upload(img200x200_2);
	
	start = std::chrono::high_resolution_clock::now();
	GPUimg350x350_1.upload(img350x350_1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Load 350x350 into GPU: " << microseconds << " microseconds" << std::endl;
	GPUimg350x350_2.upload(img350x350_2);
	
	start = std::chrono::high_resolution_clock::now();
	GPUimg1000x1000_1.upload(img1000x1000_1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Load 1000x1000 into GPU: " << microseconds << " microseconds" << std::endl;
	GPUimg1000x1000_2.upload(img1000x1000_2);
	
	start = std::chrono::high_resolution_clock::now();
	GPUimg12500x7800_1.upload(img12500x7800_1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Load 12500x7800 into GPU: " << microseconds << " microseconds" << std::endl;
	GPUimg12500x7800_2.upload(img12500x7800_2);

	
	//--------------------------------------------------------------------------------------------------------------
	// RGB to Gray into GPU
	//--------------------------------------------------------------------------------------------------------------
	std::cout << "------------------------------------------" << std::endl << std::endl;

	
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::cvtColor(GPUimg120x120_1, GPUimgGray120x120, cv::COLOR_BGR2GRAY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "BGR2GRAY 120x120 into GPU: " << microseconds << " microseconds" << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::cvtColor(GPUimg200x200_1, GPUimgGray200x200, cv::COLOR_BGR2GRAY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "BGR2GRAY 200x200 into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::cvtColor(GPUimg350x350_1, GPUimgGray350x350, cv::COLOR_BGR2GRAY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "BGR2GRAY 350x350 into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::cvtColor(GPUimg1000x1000_1, GPUimgGray1000x1000, cv::COLOR_BGR2GRAY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "BGR2GRAY 1000x1000 into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::cvtColor(GPUimg12500x7800_1, GPUimgGray12500x7800, cv::COLOR_BGR2GRAY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "BGR2GRAY 12500x7800 into GPU: " << microseconds << " microseconds" << std::endl;
	

	//--------------------------------------------------------------------------------------------------------------
	// Get a subimage from another one without copy
	//--------------------------------------------------------------------------------------------------------------
	// std::cout << "------------------------------------------" << std::endl << std::endl;

	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 100, 100));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 100x100 from 12500x7800 RGB without copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 100, 100));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 100x100 from 12500x7800 Gray without copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 250, 250));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 250x250 from 12500x7800 RGB without copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 250, 250));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 250x250 from 12500x7800 Gray without copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 1000, 1000));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 1000x1000 from 12500x7800 RGB without copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 1000, 1000));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 1000x1000 from 12500x7800 Gray without copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Get a subimage from another one with copy
	//--------------------------------------------------------------------------------------------------------------
	// std::cout << "------------------------------------------" << std::endl << std::endl;

	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 100, 100));
	temp1.copyTo(temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 100x100 from 12500x7800 RGB with copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 100, 100));
	temp2.copyTo(temp4);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 100x100 from 12500x7800 Gray with copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 250, 250));
	temp1.copyTo(temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 250x250 from 12500x7800 RGB with copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 250, 250));
	temp2.copyTo(temp4);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 250x250 from 12500x7800 Gray with copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	/*
	start = std::chrono::high_resolution_clock::now();
	temp1 = cv::cuda::GpuMat(GPUimg12500x7800_1, cv::Rect(10, 10, 1000, 1000));
	temp1.copyTo(temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 1000x1000 from 12500x7800 RGB with copy into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	temp2 = cv::cuda::GpuMat(GPUimgTemp12500x7800, cv::Rect(10, 10, 1000, 1000));
	temp2.copyTo(temp4);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SubMat 1000x1000 from 12500x7800 Gray with copy into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Resize into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimg1000x1000_1, temp1, cv::Size(990, 990));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 1000x1000 to 990x990 RGB into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimg1000x1000_1, temp2, cv::Size(990, 990));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 1000x1000 to 990x990 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimg1000x1000_1, temp1, cv::Size(950, 950));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 1000x1000 to 950x950 RGB into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimgGray1000x1000, temp2, cv::Size(950, 950));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 1000x1000 to 950x950 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimg350x350_1, temp1, cv::Size(325, 325));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 350x350 to 325x325 RGB into GPU: " << microseconds << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::resize(GPUimgGray350x350, temp2, cv::Size(325, 325));
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Resize 350x350 to 325x325 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	
	//--------------------------------------------------------------------------------------------------------------
	// Sobel with grayscale image
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;
	
	
	// Filter declaration
	/*
	cv::cuda::GpuMat gpuGrad_x, gpuGrad_y;
	cv::cuda::GpuMat abs_gpuGrad_x, abs_gpuGrad_y;

	const unsigned int ksize = 1;
	const unsigned int source_type = CV_8UC1;
	const unsigned int destination_type = CV_16S;

	cv::Ptr<cv::cuda::Filter> filterX = cv::cuda::createSobelFilter(source_type, destination_type, 1, 0, ksize, 1, cv::BORDER_DEFAULT); // use 16 bits unsigned to avoid overflow
	cv::Ptr<cv::cuda::Filter> filterY = cv::cuda::createSobelFilter(source_type, destination_type, 0, 1, ksize, 1, cv::BORDER_DEFAULT); // use 16 bits unsigned to avoid overflow
	

	
	start = std::chrono::high_resolution_clock::now();

	// gradient x direction
	filterX->apply(GPUimgGray1000x1000, gpuGrad_x);
	cv::cuda::abs(gpuGrad_x, gpuGrad_x);
	gpuGrad_x.convertTo(abs_gpuGrad_x, CV_8UC1); // CV_16S -> CV_8U

	// gradient y direction
	filterY->apply(GPUimgGray1000x1000, gpuGrad_y);
	cv::cuda::abs(gpuGrad_y, gpuGrad_y);
	gpuGrad_y.convertTo(abs_gpuGrad_y, CV_8UC1); // CV_16S -> CV_8U

	// create the output by adding the absolute gradient images of each x and y direction
	cv::cuda::addWeighted(abs_gpuGrad_x, 0.5, abs_gpuGrad_y, 0.5, 0, temp1);

	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Sobel 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();

	// gradient x direction
	filterX->apply(GPUimgGray12500x7800, gpuGrad_x);
	cv::cuda::abs(gpuGrad_x, gpuGrad_x);
	gpuGrad_x.convertTo(abs_gpuGrad_x, CV_8UC1); // CV_16S -> CV_8U

	// gradient y direction
	filterY->apply(GPUimgGray12500x7800, gpuGrad_y);
	cv::cuda::abs(gpuGrad_y, gpuGrad_y);
	gpuGrad_y.convertTo(abs_gpuGrad_y, CV_8UC1); // CV_16S -> CV_8U

	// create the output by adding the absolute gradient images of each x and y direction
	cv::cuda::addWeighted(abs_gpuGrad_x, 0.5, abs_gpuGrad_y, 0.5, 0, temp1);
	
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Sobel 12500x7800 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Laplace with grayscale image
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	
	// Filter declaration
	/*
	cv::cuda::GpuMat gpuGrad;

	const unsigned int ksize = 1;
	const unsigned int source_type = CV_8UC1;
	const unsigned int destination_type = CV_8UC1; // Destination type has to be the same type of source for now

	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLaplacianFilter(source_type, destination_type, ksize); 
	*/
	
	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray1000x1000, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Laplace 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray12500x7800, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Laplace 12500x7800 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Morphological gradient with grayscale image
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	/*
	// Filter declaration
	cv::cuda::GpuMat gpuGrad;

	const unsigned int source_type = CV_8UC1;
	const unsigned int destination_type = CV_8UC1; // Destination type has to be the same type of source for now

	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createMorphologyFilter(cv::MORPH_GRADIENT, source_type, destination_type);
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray1000x1000, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Morphological gradient 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray12500x7800, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Morphological gradient 12500x7800 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Erosion with grayscale image
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	/*
	// Filter declaration
	cv::cuda::GpuMat gpuGrad;

	const unsigned int source_type = CV_8UC1;
	const unsigned int destination_type = CV_8UC1; // Destination type has to be the same type of source for now

	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, source_type, destination_type);
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray1000x1000, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Erosion 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray12500x7800, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Erosion 12500x7800 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	/*
	start = std::chrono::high_resolution_clock::now();
	filter->apply(GPUimgGray200x200, gpuGrad);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Erosion 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Absdiff into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	
	/*
	GPUimgGray1000x1000.copyTo(temp1);

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::absdiff(GPUimgGray1000x1000, temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Absdiff 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	GPUimg350x350_1.copyTo(temp1);

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::absdiff(GPUimg350x350_1, temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Absdiff 350x350 RGB into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	GPUimgGray350x350.copyTo(temp1);

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::absdiff(GPUimgGray350x350, temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Absdiff 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Threshold into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;
	
	/*
	const double thresh = 10.0;
	const unsigned int maxVal = 255;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::threshold(GPUimgGray120x120, temp1, thresh, maxVal, cv::THRESH_BINARY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Threshold 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::threshold(GPUimgGray200x200, temp1, thresh, maxVal, cv::THRESH_BINARY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Threshold 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::threshold(GPUimgGray350x350, temp1, thresh, maxVal, cv::THRESH_BINARY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Threshold 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::threshold(GPUimgGray1000x1000, temp1, thresh, maxVal, cv::THRESH_BINARY);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Threshold 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Rotate into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;

	//const double angle = 0.68;

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::rotate(GPUimg1000x1000_1, temp1, GPUimg1000x1000_1.size(), angle);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Rotate 1000x1000 RGB into GPU: " << microseconds << " microseconds" << std::endl;
	*/
	
	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::rotate(GPUimgGray1000x1000, temp1, GPUimgGray1000x1000.size(), angle);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Rotate 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Bitwise_and into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;


	/*
	cv::cuda::threshold(GPUimgGray120x120, temp1, 10.0, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(GPUimgGray120x120, temp2, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::bitwise_and(temp1, temp2, temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Bitwise_and 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray200x200, temp1, 10.0, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(GPUimgGray200x200, temp2, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::bitwise_and(temp1, temp2, temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Bitwise_and 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray350x350, temp1, 10.0, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(GPUimgGray350x350, temp2, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::bitwise_and(temp1, temp2, temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Bitwise_and 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray1000x1000, temp1, 10.0, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(GPUimgGray1000x1000, temp2, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::bitwise_and(temp1, temp2, temp3);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Bitwise_and 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// CountNonZero into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;


	/*
	start = std::chrono::high_resolution_clock::now();
	int nonZeros = cv::cuda::countNonZero(GPUimgGray120x120);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "countNonZero 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << nonZeros << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	int nonZeros = cv::cuda::countNonZero(GPUimgGray200x200);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "countNonZero 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << nonZeros << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	int nonZeros = cv::cuda::countNonZero(GPUimgGray350x350);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "countNonZero 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << nonZeros << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	int nonZeros = cv::cuda::countNonZero(GPUimgGray1000x1000);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "countNonZero 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << nonZeros << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// MinMax into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;
	
	//double minVal, maxVal;

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::minMax(GPUimgGray120x120, &minVal, &maxVal);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "MinMax 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << minVal << " " << maxVal << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::minMax(GPUimgGray200x200, &minVal, &maxVal);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "MinMax 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << minVal << " " << maxVal << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::minMax(GPUimgGray350x350, &minVal, &maxVal);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "MinMax 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << minVal << " " << maxVal << std::endl;
	*/

	/*
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::minMax(GPUimgGray1000x1000, &minVal, &maxVal);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "MinMax 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	std::cout << minVal << " " << maxVal << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// SetTo into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;


	/*
	cv::cuda::threshold(GPUimgGray120x120, temp1, 10.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray120x120.setTo(2, temp1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SetTo 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray200x200, temp1, 12.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray200x200.setTo(2, temp1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SetTo 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray350x350, temp1, 10.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray350x350.setTo(2, temp1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SetTo 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray1000x1000, temp1, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray1000x1000.setTo(2, temp1);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "SetTo 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	//--------------------------------------------------------------------------------------------------------------
	// labelComponents into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;


	/*
	cv::cuda::threshold(GPUimgGray120x120, temp1, 128, 255, cv::THRESH_BINARY);
	cv::cuda::GpuMat mask;
	mask.create(temp1.rows, temp1.cols, CV_8UC1);
	
	cv::cuda::GpuMat components;
	components.create(temp1.rows, temp1.cols, CV_32SC1);
	std::cout << (temp1.type() == CV_8UC1) << std::endl;

	start = std::chrono::high_resolution_clock::now();
	cv::cuda::connectivityMask(cv::cuda::GpuMat(temp1), mask, cv::Scalar::all(0), cv::Scalar::all(1));
	//cv::cuda::labelComponents(mask, components);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "labelComponents 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray200x200, temp1, 12.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::labelComponents(temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "labelComponents 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray350x350, temp1, 10.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::labelComponents(temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "labelComponents 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::cuda::threshold(GPUimgGray1000x1000, temp1, 15.0, 255, cv::THRESH_BINARY);
	start = std::chrono::high_resolution_clock::now();
	cv::cuda::labelComponents(temp1, temp2);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "labelComponents 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/


	//--------------------------------------------------------------------------------------------------------------
	// Download into GPU
	//--------------------------------------------------------------------------------------------------------------
	//std::cout << "------------------------------------------" << std::endl << std::endl;


	/*
	cv::Mat tmp;
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray120x120.download(tmp);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Download 120x120 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::Mat tmp;
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray200x200.download(tmp);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Download 200x200 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::Mat tmp;
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray350x350.download(tmp);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Download 350x350 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::Mat tmp;
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray1000x1000.download(tmp);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Download 1000x1000 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	/*
	cv::Mat tmp;
	start = std::chrono::high_resolution_clock::now();
	GPUimgGray12500x7800.download(tmp);
	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Download 12500x7800 Gray into GPU: " << microseconds << " microseconds" << std::endl;
	*/

	//--------------------------------------------------------------------------------------------------------------
	// Download from GPU an image and save on disk
	//--------------------------------------------------------------------------------------------------------------
	// cv::Mat dest;
	// gpuImg.download(dest);
	// cv::imwrite("prueba.png", dest);


	//--------------------------------------------------------------------------------------------------------------
	// END
	//--------------------------------------------------------------------------------------------------------------
	std::cout << "------------------------------------------" << std::endl << std::endl;
	
  system("pause");
  return 0;
}

