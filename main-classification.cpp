#include "NeuralNetwork.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>
#include "DataSet.hpp"
using namespace std;

void DrawTraining(LKY::NeuralNetwork _nn, int maxEpochs, int currentEpochs, const vector<vector<double>>& displayData)
{   //size_t numItems = 80;
    string strPngName = "png/訓練途中" + to_string(currentEpochs) + ".png";
    string strPutText = "Epoch:"+to_string(currentEpochs)+"/"+to_string(maxEpochs)+"  Err:" + to_string(_nn.GetTrainError().back());

    //cv::imwrite(strPngName.c_str(),DrawData("訓練途中", displayData, strPutText));
    Draw2DClassificationData("訓練途中", displayData, _nn, strPutText);
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
    vector<vector<double>> trainData = Make2DBinaryTrainingData();

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(2, 16, 2, statrTime);
    nn.SetClassification(); //設定為分類器
    //nn.SetActivation(new ReLU());
    //nn.SetLossFunction(new CrossEntropy());
    //nn.ShowWeights();//訓練前

    int maxEpochs = 100000;
    double learnRate = 0.00005;
    double momentum  = 0.00005;
    nn.eventInTraining = DrawTraining;//將包有視覺化的事件傳入
    nn.Train(trainData, maxEpochs, learnRate, momentum);

    cout << "\nEnd demo\n";

    auto endTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "execute time= " << endTime - statrTime << "ms"<< std::endl;
    
    //cv::waitKey(30);
    fgetc(stdin);
}