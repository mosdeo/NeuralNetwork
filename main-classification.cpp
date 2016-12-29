#include "NeuralNetwork.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>
#include "DrawData.hpp"
#include "classifyCircleData.hpp"
using namespace std;

vector<vector<double>> Make2DBinaryTrainingData(int numTariningData = 40)
{
    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numTariningData, vector<double>(3));
    LKY::NeuralNetwork::Random rnd = LKY::NeuralNetwork::Random(0);

    //產生兩個類別的資料點
    double A_centerX = 1.5,   A_centerY = 1.5;
    double B_centerX = -1.5, B_centerY = -1.5;
    double noiseRate = 7;

    for (size_t i = 0; i < trainData.size(); ++i)
    {
        if(i>trainData.size()/2)
        {
            trainData[i][0] = A_centerX + noiseRate*(rnd.NextDouble()-0.5);
            trainData[i][1] = A_centerY + noiseRate*(rnd.NextDouble()-0.5);
            trainData[i].back() = 1;
        }
        else
        {
            trainData[i][0] = B_centerX + noiseRate*(rnd.NextDouble()-0.5);
            trainData[i][1] = B_centerY + noiseRate*(rnd.NextDouble()-0.5);
            trainData[i].back() = -1;
        }
    }

    return trainData;
}

void DrawTraining(LKY::NeuralNetwork _nn, int maxEpochs, int currentEpochs)
{   //size_t numItems = 80;
    vector<vector<double>> testData = classifyCircleData();//Make2DBinaryTrainingData();

    string strPngName = "png/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn.GetTrainError().back());

    //cv::imwrite(strPngName.c_str(),DrawData("訓練途中", testData, strPutText));
    Draw2DClassificationData("訓練途中", testData, _nn, strPutText);
    //fgetc(stdin);
}

int main(int argc, char* argv[])
{
    auto statrTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "statrTime= " << statrTime << std::endl;

    cout << "Begin neural network classification demo" << endl;
    cout << "Goal is to discriminate the +1 -1" << endl;

    int numTariningData = 80;
    cout << "Programmatically generating " + to_string(numTariningData) + " training data items" << endl;

    //make 2*numItems 2D vector
    //產生兩個類別的資料點
    vector<vector<double>> trainData = classifyCircleData();//Make2DBinaryTrainingData();

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(2, 8, 2, statrTime);
    nn.SetClassification(); //設定為分類器
    //nn.ShowWeights();//訓練前

    int maxEpochs = 100000;
    double learnRate = 0.0012;
    double momentum  = 0.0003;
    nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入
    nn.Train(trainData, maxEpochs, learnRate, momentum);

    cout << "\nEnd demo\n";

    auto endTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "execute time= " << endTime - statrTime << "ms"<< std::endl;
    
    //cv::waitKey(30);
    fgetc(stdin);
}