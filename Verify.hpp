#include "../NeuralNetwork/NeuralNetwork.hpp"
#include <iostream>
#include <vector>

void Verify(LKY::NeuralNetwork& nn, vector<vector<double>> verifyData)
{
    //求 皮爾森積差
    double CORR = 0;
    double ActualPSPIAvg = 0, PredictedPSPIAvg = 0;
    double COVxy = 0, Sx=0, Sy = 0;

    for (size_t i = 0; i < verifyData.size(); i++)
    {//算出實際和預測的平均值
        double Predicted = nn.ComputeOutputs(verifyData[i])[0];
        double Actual = verifyData[i][verifyData[i].size() - 1];
        PredictedPSPIAvg += Predicted;
        ActualPSPIAvg += Actual;
    }
    PredictedPSPIAvg /= verifyData.size();
    ActualPSPIAvg /= verifyData.size();

    for (size_t i = 0; i < verifyData.size(); i++)
    {//求差
        double Xerr = 0, Yerr = 0;
        Xerr = verifyData[i][verifyData[i].size() - 1] - ActualPSPIAvg;
        Yerr =nn.ComputeOutputs(verifyData[i])[0] - PredictedPSPIAvg;
        COVxy += Xerr * Yerr;
        Sx += pow(Xerr, 2);
        Sy += pow(Yerr, 2);
    }
    CORR = COVxy/pow(Sx*Sy, 0.5);
    string strCORR = "CORR = " + to_string(CORR);
    cout << strCORR << endl;
    //結束


    //求SetB MSE
    double MSE = 0;
    for (size_t i = 0; i < verifyData.size(); i++)
    {
        //nn.ComputeOutputs只看建構子的numInput讀資料長度，所以inputVector[i]最後一項y-data會自動被忽略。
        double Predicted = nn.ComputeOutputs(verifyData[i])[0];
        double Actual = verifyData[i][verifyData[i].size()-1];
        MSE += pow(Actual - Predicted, 2);
    }
    MSE = MSE / verifyData.size();
    cout << "MSE:" << MSE << endl;
    //結束
}