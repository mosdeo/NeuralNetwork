#include "NeuralNetwork.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>
#include "DrawData.hpp"
using namespace std;

int main()
{
    auto statrTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "statrTime= " << statrTime << std::endl;

    cout << "Begin neural network regression demo" << endl;
    cout << "Goal is to predict the sin(x)" << endl;

    int numTariningData = 80;
    cout << "Programmatically generating " + to_string(numTariningData) + " training data items" << endl;

    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numTariningData, vector<double>(2));

    LKY::NeuralNetwork::Random rnd = LKY::NeuralNetwork::Random();

    //產生一個周期內的80個sin取樣點
    for (int i = 0; i < numTariningData; ++i)
    {
        double x = 2*M_PI*rnd.NextDouble(); // [0 to 2PI]
        double sx = cos(2*x);
        trainData[i][0] = x;
        trainData[i][1] = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }
    cout << endl;
    cout << "Training data:" << endl;

    DrawData("訓練資料",trainData,"Training Data");
    //cv::waitKey(30);
    //fgetc(stdin);
    cv::destroyWindow("訓練資料");

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(1, 4, 1, 0);
    nn.isVisualizeTraining = false;
    //nn.ShowWeights();//訓練前

    int maxEpochs = 2000;
    double learnRate = 0.007;
    double momentum = 0.005;
    nn.Train(trainData, maxEpochs, learnRate, momentum);
    //nn.ShowWeights();//訓練後

    vector<double> y;
    y = nn.ComputeOutputs(vector<double>(numTariningData, M_PI));
    cout << "\nActual sin(PI)       =  0.0   Predicted =  " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numTariningData, M_PI/2.0));
    cout << "\nActual sin(PI / 2)   =  1.0   Predicted =  " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numTariningData, 3*M_PI/2.0));
    cout << "\nActual sin(3*PI / 2) = -1.0   Predicted = " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numTariningData, 6*M_PI));
    cout << "\nActual sin(6*PI)     =  0.0   Predicted =  " + to_string(y[0]) << endl;

    cout << "\nEnd demo\n";

    auto endTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "execute time= " << endTime - statrTime << "ms"<< std::endl;
    
    cv::waitKey(30);
    fgetc(stdin);
}