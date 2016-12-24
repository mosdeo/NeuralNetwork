#include "../NeuralNetwork/NeuralNetwork.hpp"
#include <iostream>
#include <vector>

void Testing(LKY::NeuralNetwork& nn, const vector<vector<double>>& testData, double& CORR, double& MSE)
{
    //求 皮爾森積差
    CORR = 0;
    double ActualPSPIAvg = 0, PredictedPSPIAvg = 0;
    double COVxy = 0, Sx= 0, Sy = 0;

    for (size_t i = 0; i < testData.size(); i++)
    {//算出實際和預測的平均值
        double Predicted = nn.ComputeOutputs(testData[i])[0];
        double Actual = *(testData[i].back());
        PredictedPSPIAvg += Predicted;
        ActualPSPIAvg += Actual;
    }
    PredictedPSPIAvg /= testData.size();
    ActualPSPIAvg /= testData.size();

    for (size_t i = 0; i < testData.size(); i++)
    {//求差
        double Xerr = 0, Yerr = 0;
        Xerr = *(testData[i].back()) - ActualPSPIAvg;
        Yerr =nn.ComputeOutputs(testData[i])[0] - PredictedPSPIAvg;
        COVxy += Xerr * Yerr;
        Sx += pow(Xerr, 2);
        Sy += pow(Yerr, 2);
    }
    CORR = COVxy/pow(Sx*Sy, 0.5);
    string strCORR = "CORR = " + to_string(CORR);
    cout << strCORR << endl;
    //結束

    //求MSE
    MSE = 0;
    for (size_t i = 0; i < testData.size(); i++)
    {
        //nn.ComputeOutputs 只看建構子的 numInput讀資料長度，所以 inputVector[i] 最後一項y-data會自動被忽略。
        double Predicted = nn.ComputeOutputs(testData[i])[0];
        double Actual = *(testData[i].back());
        MSE += pow(Actual - Predicted, 2);
    }
    MSE = MSE / testData.size();
    cout << "MSE = " << MSE << endl;
    //結束
}
