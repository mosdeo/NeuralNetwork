#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>
#include "DrawData.hpp"
using namespace std;

int main()
{  
    cv::VideoWriter* ptrWriter = new cv::VideoWriter("Sin(x) NN Learning.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, cv::Size(320, 240));
    if(!ptrWriter->isOpened())
    {
        cerr << "Could not open the output video file for write\n";
        return EXIT_FAILURE;
    }

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
        double x = 6.4 * rnd.NextDouble(); // [0 to 2PI]
        double sx = sin(x);
        trainData[i][0] = x;
        trainData[i][1] = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }
    cout << endl;
    cout << "Training data:" << endl;

    DrawData("訓練資料",trainData);
    cv::waitKey(30);
    //fgetc(stdin);
    cv::destroyWindow("訓練資料");

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(1, 12, 1, 0);
    nn.ShowWeights();//訓練前

    int maxEpochs = 1000;
    double learnRate = 0.05;
    double momentum = 0.005;
    nn.Train(trainData, maxEpochs, learnRate, momentum);
    nn.ShowWeights();//訓練後

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
    
    cv::waitKey(30);
    free(ptrWriter);
    //fgetc(stdin);
}