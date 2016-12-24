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
        double sx = sin(x);
        trainData[i][0] = x;
        trainData[i][1] = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }
    cout << endl;
    cout << "Training data:" << endl;

    // DrawData("訓練資料",trainData,"Training Data");
    // cv::waitKey(3000);
    // fgetc(stdin);
    // cv::destroyWindow("訓練資料");

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(1, 8, 1, statrTime);
    //nn.isVisualizeTraining = true;
    //double dWeights[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5};
    //vector<double> verifyWeights(dWeights,dWeights+sizeof(dWeights)/sizeof(double));
    //nn.SetWeights(verifyWeights);
    nn.ShowWeights();//訓練前

    int maxEpochs = 1000;
    double learnRate = 0.0005;
    double momentum = 0.000005;
    nn.Train(trainData, maxEpochs, learnRate, momentum);
    nn.ShowWeights();//訓練後

    // double dInputs[] = {-100, 100};
    // vector<double> vecInputs(dInputs, dInputs+sizeof(dInputs)/sizeof(double));
    // vector<double> y;
    // y = nn.ComputeOutputs(vecInputs);
    // cout << "\ny[0]" + to_string(y[0]) << endl;
    // cout << "\ny[1]" + to_string(y[1]) << endl;

    cout << "\nEnd demo\n";

    auto endTime = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
    cout << "execute time= " << endTime - statrTime << "ms"<< std::endl;
    
    //cv::waitKey(30);
    fgetc(stdin);
}