#ifndef _DARWDATA_HPP_
#define _DARWDATA_HPP_
#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>
using namespace std;

cv::Mat Draw2DClassifierData(string strWindowName ,vector<vector<double>> XYData, LKY::NeuralNetwork& _nn, string strPutText="LKY",
    double Xmin = -10, double Xmax = 10, double Ymin = -10, double Ymax = 10)
{
    cv::Size canvasSize(400, 400); //畫布大小
    cv::Mat canvas(canvasSize, CV_8UC3, cv::Scalar(0));//產生畫布

    //計算修正參數
    double XscaleRate = canvasSize.width/(Xmax-Xmin);
    double YscaleRate = canvasSize.height/(Ymax-Ymin);
    
    //寫入機率密度分佈
    for (int x = 0 ; x < canvasSize.width ; x++)
    {
        for (int y = 0 ; y < canvasSize.height ; y++)
        {
            double resvY = y/YscaleRate + Ymin; 
            double resvX = x/XscaleRate + Xmin;
            vector<double> result = _nn.ComputeOutputs(vector<double>{resvX,resvY});

            if(result[0] < result[1])
            {
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(255*2*result[0], 255, 255*2*result[0]);
            }
            if(result[1] < result[0])
            {
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(255*2*result[1], 255*2*result[1], 255);
            }
        }
    }

    //寫入每個資料點畫素
    for (size_t i = 0 ; i < XYData.size() ; i++)
    {
        int newY = YscaleRate*(XYData[i][1]-Ymin);
        int newX = XscaleRate*(XYData[i][0]-Xmin);

        cv::Scalar circleColor;
        if(XYData[i].back()== 1){circleColor = cv::Scalar(0, 0, 255);}//鮮紅色
        if(XYData[i].back()==-1){circleColor = cv::Scalar(0, 205, 0);}//深綠色onst

        const int radius = 5;
        const int thickness = 2;
        cv::circle(canvas, cv::Point(newY, newX), radius, circleColor, thickness);
    }

    cv::putText(canvas,strPutText.c_str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com", cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5, cv::Scalar(0x00));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};


cv::Mat Draw2DRegressionData(string strWindowName ,vector<vector<double>> XYData, string strPutText="LKY", double Xmin = 0, double Xmax = 6.4, double Ymin = -3, double Ymax = 3)
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
        canvas.at<unsigned char>(newY, newX)=255;//pixel write
    }

    cv::dilate(canvas, canvas, cv::Mat()); //使畫素膨脹

    cv::putText(canvas,strPutText.c_str(),cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(255));
    cv::putText(canvas,"Lin Kao-Yuan, mosdeo@gmail.com",cv::Point(20,canvas.rows-20), cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(127));

    //顯示
    cv::imshow(strWindowName.c_str(), canvas);
    cv::waitKey(1);

    return canvas;
};

#endif