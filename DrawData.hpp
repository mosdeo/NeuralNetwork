#ifndef _DARWDATA_HPP_
#define _DARWDATA_HPP_
#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

cv::Mat Draw2DRegressionData(string strWindowName ,vector<vector<double>> XYData, string strPutText="LKY", double Xmin = 0, double Xmax = 6.4, double Ymin = -3, double Ymax = 3)
{
    cv::Size canvasSize(640, 480); //畫布大小
    cv::Mat canvas(canvasSize, CV_8U, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    double Xshitf = -Xmin;
    double Yshitf = -Ymin;
    
    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]+Yshitf);
        int newX = XscaleRate*(XYData[i][0]+Xshitf);
        //printf("newX = %d, newY = %d\n", newX, newY);
        canvas.at<unsigned char>(newY, newX)=255;//pixel write
    }

    //cv::resize(canvas, canvas, cv::Size(640, 480), cv::INTER_NEAREST);
    cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(127));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};


cv::Mat Draw2DClassifierData(string strWindowName ,vector<vector<double>> XYData, LKY::NeuralNetwork& _nn, string strPutText="LKY", double Xmin = -10, double Xmax = 10, double Ymin = -5, double Ymax = 5)
{
    cv::Size canvasSize(640, 480); //畫布大小
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    double Xshitf = -Xmin;
    double Yshitf = -Ymin;
    
    //寫入機率密度分佈
    for (int x = 0 ; x < canvasSize.width ; x++)
    {
        for (int y = 0 ; y < canvasSize.height ; y++)
        {
            double resvY = y/YscaleRate - Yshitf; //YscaleRate*(XYData[i][1]+Yshitf);
            double resvX = x/XscaleRate - Xshitf; //XscaleRate*(XYData[i][0]+Xshitf);
            vector<double> result = _nn.ComputeOutputs(vector<double>{resvX,resvY});

            if(result[0] < result[1])
            {
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(255*2*result[0], 255, 255*2*result[0]);
            }
            if(result[1] < result[0])
            {
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(255*2*result[1], 255*2*result[1], 255);
            }

            // canvas.at<cv::Vec3b>(y, x)[1]=255*result[1];//Green pixel write
            // canvas.at<cv::Vec3b>(y, x)[2]=255*result[0];//Red   pixel write
        }
    }

    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]+Yshitf);
        int newX = XscaleRate*(XYData[i][0]+Xshitf);
        //printf("newX = %d, newY = %d\n", newX, newY);

        // if(XYData[i].back()== 1)canvas.at<cv::Vec3b>(newY, newX) = cv::Vec3b(0, 0, 255);//pure Red
        // if(XYData[i].back()==-1)canvas.at<cv::Vec3b>(newY, newX) = cv::Vec3b(0, 255, 0);//pure Green

        cv::Scalar color;
        if(XYData[i].back()== 1)
        {
            color = cv::Scalar(0, 0, 255);
        }//pure Red
        else if(XYData[i].back()==-1)
        {
            color = cv::Scalar(0, 255, 0);
        }//pure Green

        cv::circle(canvas, cv::Point(newY, newX), 5, color,2);
    }

    //cv::resize(canvas, canvas, cv::Size(640, 480), cv::INTER_NEAREST);
    //cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255,255,255));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255,255,255));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

#endif