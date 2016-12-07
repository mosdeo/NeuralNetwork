#ifndef _DARWDATA_HPP_
#define _DARWDATA_HPP_

#include <opencv2/opencv.hpp>
using namespace std;

cv::Mat DrawData(string strWindowName ,vector<vector<double>> XYData, string strPutText="LKY", double Xmin = 0, double Xmax = 6.4, double Ymin = -3, double Ymax = 3)
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

#endif