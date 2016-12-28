#ifndef _DARWDATA_HPP_
#define _DARWDATA_HPP_
#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

cv::Mat Draw2DRegressionData(string strWindowName ,vector<vector<double>> XYData, string strPutText="LKY", double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(640, 480); //畫布大小
    cv::Mat canvas(canvasSize, CV_8U, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);
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


cv::Mat Draw2DClassifierData(string strWindowName ,vector<vector<double>> XYData, LKY::NeuralNetwork& _nn, string strPutText="LKY", double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(800, 800); //畫布大小
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //寫入機率密度分佈
    for (int pixelX = 0 ; pixelX < canvasSize.width ; pixelX++)
    {
        for (int pixelY = 0 ; pixelY < canvasSize.height ; pixelY++)
        {
            double resvY = pixelY/YscaleRate + Ymin;
            double resvX = pixelY/XscaleRate + Xmin;
            vector<double> result = _nn.ComputeOutputs(vector<double>{resvX,resvY});

            if(result[0] < result[1])
            {
                canvas.at<cv::Vec3b>(pixelY, pixelX) = cv::Vec3b(255*2*result[0], 255, 255*2*result[0]);
            }
            if(result[1] < result[0])
            {
                canvas.at<cv::Vec3b>(pixelY, pixelX) = cv::Vec3b(255*2*result[1], 255*2*result[1], 255);
            }
        }
    }

    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int pixelY = YscaleRate*(XYData[i][1]-Ymin);
        int pixelX = XscaleRate*(XYData[i][0]-Xmin);
        
        cv::Scalar color;
        if(XYData[i].back()== 1)
        {
            color = cv::Scalar(0, 0, 255);
        }//pure Red
        else if(XYData[i].back()==-1)
        {
            color = cv::Scalar(0, 255, 0);
        }//pure Green

        cv::circle(canvas, cv::Point(pixelX, pixelY), 5, color,2);
    }

    //cv::resize(canvas, canvas, cv::Size(640, 480), cv::INTER_NEAREST);
    //cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

#endif