/*	CS585_Lab2.cpp
 *	CS585 Image and Video Computing Fall 2014
 *	Lab 2
 *	--------------
 *	This program introduces the following concepts:
 *		a) Reading a stream of images from a webcamera, and displaying the video
 *		b) Skin color detection
 *		c) Background differencing
 *		d) Visualizing motion history
 *	--------------
 */

//#include "stdafx.h"


//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <vector>
#include <stdio.h>



using namespace cv;
using namespace std;

int  thresh = 100;
int max_thresh = 255;
RNG rng(12345);

//function declarations

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);

/**
	Function that returns the maximum of 3 integers
	@param a first integer
	@param b second integer
	@param c third integer
 */
int myMax(int a, int b, int c);

/**
	Function that returns the minimum of 3 integers
	@param a first integer
	@param b second integer
	@param c third integer
 */
int myMin(int a, int b, int c);

/**
	Function that detects whether a pixel belongs to the skin based on RGB values
	@param src The source color image
	@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
 */
void mySkinDetect(Mat& src, Mat& dst);

/**
	Function that does frame differencing between the current frame and the previous frame
	@param src The current color image
	@param prev The previous color image
	@param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
 and previous image are not the same
 */
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

/**
	Function that accumulates the frame differences for a certain number of pairs of frames
	@param mh Vector of frame difference images
	@param dst The destination grayscale image to store the accumulation of the frame difference images
 */
void myMotionEnergy(Vector<Mat> mh, Mat& dst);

void myHistogram(Mat& src);

int main()
{
    
    //----------------
    //a) Reading a stream of images from a webcamera, and displaying the video
    //----------------
    // For more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    // open the video camera no. 0
    VideoCapture cap(0);
    
    // if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    
    //create a window called "MyVideoFrame0"
    namedWindow("MyVideo0",WINDOW_AUTOSIZE);
    Mat frame0;
    
    // read a new frame from video
    bool bSuccess0 = cap.read(frame0);
    
    //if not successful, break loop
    if (!bSuccess0)
    {
        cout << "Cannot read a frame from video stream" << endl;
    }
    
    //show the frame in "MyVideo" window
//    imshow("MyVideo0", frame0);
    
    //create a window called "MyVideo"
    namedWindow("MyVideo",WINDOW_AUTOSIZE);
    namedWindow("MyVideoMH",WINDOW_AUTOSIZE);
    
    vector<Mat> myMotionHistory;
    Mat fMH1, fMH2, fMH3;
    fMH1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
    fMH2 = fMH1.clone();
    fMH3 = fMH1.clone();
    myMotionHistory.push_back(fMH1);
    myMotionHistory.push_back(fMH2);
    myMotionHistory.push_back(fMH3);
    
    while (1)
    {
        // read a new frame from video
        Mat frame;
        bool bSuccess = cap.read(frame);
        
        //if not successful, break loop
        if (!bSuccess)
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        
        // destination frame
        Mat frameDest;
        frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
        
        Mat frameDest1;
        frameDest1 = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        Mat frameDest2;
        frameDest2 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
        ////----------------
        ////	b) Skin color detection
        ////----------------
        mySkinDetect(frame, frameDest1);
        mySkinDetect(frame0, frameDest2);
        
        ////----------------
        ////	c) Background differencing
        ////----------------
        
        
        //        call myFrameDifferencing function
        myFrameDifferencing(frameDest1, frameDest2, frameDest);
        imshow("MyVideo", frameDest);
        myMotionHistory.erase(myMotionHistory.begin());
        myMotionHistory.push_back(frameDest);
        Mat myMH = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
        
        ////----------------
        ////  d) Visualizing motion history
        ////----------------
        //
        //  call myMotionEnergy function
        myMotionEnergy(myMotionHistory, myMH);
        
//        myHistogram(myMH);
        Mat binary;
        vector < vector<Point2i > > blobs;
//        Mat binary;
        
        cv::threshold(myMH, binary, 0.0, 1.0, cv::THRESH_BINARY);
        
        FindBlobs(binary, blobs);

        
        imshow("MyVideoMH", myMH); //show the frame in "MyVideo" window
        frame0 = frame;
        //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (waitKey(30) == 27)
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        
    }
    cap.release();
    return 0;
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
    int m = a;
    (void)((m < b) && (m = b));
    (void)((m < c) && (m = c));
    return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
    int m = a;
    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));
    return m;
}



//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
    //Surveys of skin color modeling and detection techniques:
    //Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    //Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i,j); //Vec3b is a vector of 3 uchar (unsigned character)
            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
            if ((R > 95 && G > 40 && B > 20) && (myMax(R,G,B) - myMin(R,G,B) > 15) && (abs(R-G) > 15) && (R > G) && (R > B)){
                dst.at<uchar>(i,j) = 255;
            }
        }
    }
}

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
    //For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    //For more information on how to use background subtraction methods: http://docs.opencv.org/trunk/doc/tutorials/video/background_subtraction/background_subtraction.html
    absdiff(prev, curr, dst);
    Mat gs = dst.clone();
//    cvtColor(dst, gs, CV_BGR2GRAY);
//    dst = gs > 50;
    Vec3b intensity = dst.at<Vec3b>(100,100);
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(Vector<Mat> mh, Mat& dst) {
    Mat mh0 = mh[0];
    Mat mh1 = mh[1];
    Mat mh2 = mh[2];
    
    for (int i = 0; i < dst.rows; i++){
        for (int j = 0; j < dst.cols; j++){
            if (mh0.at<uchar>(i,j) == 255 || mh1.at<uchar>(i,j) == 255 ||mh2.at<uchar>(i,j) == 255){
                dst.at<uchar>(i,j) = 255;
            }
        }
    }
}



void myHistogram(Mat& src)
{
//    int min_x = src.cols;
//    int max_x = 0;
//    int min_y = src.rows;
//    int max_y = 0;
//    
//    int step = 0;
//    
//    for (int i = 0; i < src.rows; i+=step){
//        for (int j = 0; j < src.cols; j+=step){
//            if (src.at<uchar>(i,j) == 255){

                
//                if(min_x>i) min_x = i;
//                if(max_x<i) max_x = i;
//                if(min_y>j) min_y = j;
//                if(max_y<j) max_y = j;
                
                
               
//            }
//        }
//    }
    
//    cout << "rows " << src.rows << " cols " << src.cols<<endl;
//
//    Point x(min_x,min_y);
//    Point y(max_x,max_y);
//    Rect rect(x,y);
//    rectangle(src, x, y, (0,0,255));
}

//void findCont(Mat& src_gray)
//{
//    Mat canny_output;
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;
//    int  thresh = 100;
//    int max_thresh = 255;
//    
//    /// Detect edges using canny
//    Canny( src_gray, canny_output, CvHistogram::thresh, thresh*2, 3 );
//    /// Find contours
//    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//    
//    /// Draw contours
//    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
//    for( int i = 0; i< contours.size(); i++ )
//    {
//        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
//    }
//    
//    /// Show in a window
//    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//    imshow( "Contours", drawing );
//}


//void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
//{
//    blobs.clear();
//    
//    // Fill the label_image with the blobs
//    // 0  - background
//    // 1  - unlabelled foreground
//    // 2+ - labelled foreground
//    
//    cv::Mat label_image;
//    binary.convertTo(label_image, CV_32SC1);
//    
//    int label_count = 2; // starts at 2 because 0,1 are used already
//    
//    for(int y=0; y < label_image.rows; y++) {
//        int *row = (int*)label_image.ptr(y);
//        for(int x=0; x < label_image.cols; x++) {
//            if(row[x] != 1) {
//                continue;
//            }
//            
//            cv::Rect rect;
//            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
//            
//            std::vector <cv::Point2i> blob;
//            
//            for(int i=rect.y; i < (rect.y+rect.height); i++) {
//                int *row2 = (int*)label_image.ptr(i);
//                for(int j=rect.x; j < (rect.x+rect.width); j++) {
//                    if(row2[j] != label_count) {
//                        continue;
//                    }
//                    
//                    blob.push_back(cv::Point2i(j,i));
//                }
//            }
//            
//            blobs.push_back(blob);
//            
//            label_count++;
//        }
//    }
//    
//    std::cout << "Number of blobs" << label_count;
//}

