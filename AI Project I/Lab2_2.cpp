/*	CS440_P1.cpp
	@author: Leen AlShenibr, Veena Dali, Yeskendir Kazmurat
    
    some of the code was adapted from Kyle Hounslow (motionTracking.cpp December 2013)
 
	Programming Assignment 1
	--------------
	This program:
		 Recognizes gestures (i.e. throwing, petting, and wiping) and creates a graphical display in response to the gestures
	--------------
*/

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

//dimensions of 2-D array used for frame
const static int X_ARR = 2;
const static int Y_ARR = 50;

//Function Declarations

//function that returns the maximum of 3 integers
int myMax(int a, int b, int c);

//function that returns the minimum of 3 integers
int myMin(int a, int b, int c);

//function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst);

//function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

//function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(Vector<Mat> mh, Mat& dst);

//function used to find the min and max in the 2D array
int minX(int poiArr[X_ARR][Y_ARR]);
int maxX(int poiArr[X_ARR][Y_ARR]);
int minY(int poiArr[X_ARR][Y_ARR]);
int maxY(int poiArr[X_ARR][Y_ARR]);

//2D array used for gesture identification
int pointArray[X_ARR][Y_ARR];

        //const static int BLUR_SIZE = 10;

//we'll have just one object to search for and keep track of its position
int theObject[2] = {0,0};
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0,0,0,0);

//int to string helper function
string intToString(int number){

    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed){
    bool objectDetected = false;
    Mat temp;
    thresholdImage.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours
    
    Scalar color(0, 255, 0);
    Mat tempImg= Mat::zeros(temp.rows, temp.cols, CV_8UC1);
    drawContours(tempImg, contours, -1, color, 3);
    
    imshow("contours", tempImg);

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)objectDetected=true;
    else objectDetected = false;

    if(objectDetected){
        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));
        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));
        int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }
    //make some temp x and y variables so we dont have to type out so much
    int x = theObject[0];
    int y = theObject[1];
    
    //Update position in array
    if (position > 50)
    {
        position = position % 50;
        
        //Call Veena's code
    }
    pointArray[position][0] = x;
    pointArray[position][1] = y;
    position++;

    //draw some crosshairs around the object
    circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

    //write the position of the object to the screen
    putText(cameraFeed,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);
}


int main()
{
    
    //reading a stream of images from a webcamera and displaying the video
    VideoCapture cap(0);
    
    // if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    
    namedWindow("MyVideo0",WINDOW_AUTOSIZE);
    Mat frame0;
    
    //read a new frame from video
    bool bSuccess0 = cap.read(frame0);
    
    //if not successful, break loop
    if (!bSuccess0)
    {
        cout << "Cannot read a frame from video stream" << endl;
    }
    
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
        
        //destination frame
        Mat frameDest;
        frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
        
        Mat frameDest1;
        frameDest1 = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        Mat frameDest2;
        frameDest2 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
        
        //Blurs image to get less noise in the image
        blur(frame,frame,cv::Size(10,10));
        blur(frame0,frame0,cv::Size(10,10));
       
        //Skin color detection
        mySkinDetect(frame, frameDest1);
        mySkinDetect(frame0, frameDest2);
        
        //background differencing
        myFrameDifferencing(frameDest1, frameDest2, frameDest);
        imshow("MyVideo", frameDest);
        myMotionHistory.erase(myMotionHistory.begin());
        myMotionHistory.push_back(frameDest);
        Mat myMH = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
        
        //Visualizing motion history
        myMotionEnergy(myMotionHistory, myMH);
        
        imshow("MyVideoMH", myMH); //show the frame in "MyVideo" window
        frame0 = frame;
        
        searchForMovement(frameDest,myMH);
        
        imshow("Frame", myMH);
        
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

void minX() {
    
    int min = 5000000;
    int currentMin[2]= {0, 0};
    
    for(int i = 0; i < X_ARR; i++)
    {
        if (pointArray[i][0] < min)
        {
            min = pointArray[i][0];
            currentMin[0] = pointArray[i][0];
            currentMin[1] = pointArray[i][1];
        }
    }
    
    
    minXPoint[0] = currentMin[0];
    minXPoint[1] = currentMin[1];
    
}

void maxX() {
    
    int max = -5000000;
    int currentMax[2]= {0, 0};
    
    for(int i = 0; i < X_ARR; i++)
    {
        if (pointArray[i][0] > max)
        {
            max = pointArray[i][0];
            currentMax[0] = pointArray[i][0];
            currentMax[1] = pointArray[i][1];
        }
    }
    
    
    maxXPoint[0] = currentMax[0];
    maxXPoint[1] = currentMax[1];
    
    
}


void minY() {
    
    int min = 5000000;
    int currentMin[2]= {0, 0};
    
    for(int i = 0; i < Y_ARR; i++)
    {
        if (pointArray[i][1] < min)
        {
            min = pointArray[i][1];
            currentMin[0] = pointArray[i][0];
            currentMin[1] = pointArray[i][1];
        }
    }
    
    minYPoint[0] = currentMin[0];
    minYPoint[1] = currentMin[1];
    
    
}

void maxY() {
    
    int max = -5000000;
    int currentMax[2]= {0, 0};
    
    for(int i = 0; i < Y_ARR; i++)
    {
        if (pointArray[i][1] > max)
        {
            max = pointArray[i][1];
            currentMax[0] = pointArray[i][0];
            currentMax[1] = pointArray[i][1];
        }
    }
    
    maxYPoint[0] = currentMax[0];
    maxYPoint[1] = currentMax[1];
    
    
}


