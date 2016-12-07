#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>
#include "DrawData.hpp"
using namespace std;

int main()
{
    // cv::Mat img = cv::imread("西屯藍天.jpg");
    // cv::Size sizeVGA(640,480);
    // cv::resize(img, img, sizeVGA);
    // cv::imshow("test img",img);
    // cv::waitKey(0);
    // cv::destroyWindow("test img");

    cout << "Begin neural network regression demo" << endl;
    cout << "Goal is to predict the sin(x)" << endl;

    int numItems = 200;
    cout << "Programmatically generating " + to_string(numItems) + " training data items" << endl;

    //make 2*numItems 2D vector
    vector<vector<double>> trainData(numItems, vector<double>(2));

    LKY::NeuralNetwork::Random rnd = LKY::NeuralNetwork::Random();

    //產生一個周期內的80個sin取樣點
    for (int i = 0; i < numItems; ++i)
    {
        double x = 6.4 * rnd.NextDouble(); // [0 to 2PI]
        double sx = sin(x);
        trainData[i][0] = x;
        trainData[i][1] = sx;
        //printf("x=%lf, sx=%lf\n", x, sx);
    }
    cout << endl;
    cout << "Training data:" << endl;
    //Show.ShowMatrix(trainData, 3, 4, true);

    DrawData(trainData);
    cv::waitKey(0);

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(1, 12, 1, 0);

    //訓練前
    vector<double> weights = nn.GetWeights();
    std::cout << "weights:" << endl;
    for(auto const &n : weights)
    {
        cout << ' ' << n;
    }cout << endl;


    int maxEpochs = 10000;
    double learnRate = 0.005;
    double momentum = 0.005;
    nn.Train(trainData, maxEpochs, learnRate, momentum);

    //訓練後
    weights = nn.GetWeights();
    std::cout << "weights:" << endl;
    for(auto const &n : weights)
    {
        cout << ' ' << n;
    }cout << endl;

    vector<double> y;
    y = nn.ComputeOutputs(vector<double>(numItems, M_PI));
    cout << "\nActual sin(PI)       =  0.0   Predicted =  " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numItems, M_PI/2.0));
    cout << "\nActual sin(PI / 2)   =  1.0   Predicted =  " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numItems, 3*M_PI/2.0));
    cout << "\nActual sin(3*PI / 2) = -1.0   Predicted = " + to_string(y[0]) << endl;

    y = nn.ComputeOutputs(vector<double>(numItems, 6*M_PI));
    cout << "\nActual sin(6*PI)     =  0.0   Predicted =  " + to_string(y[0]) << endl;

    cout << "\nEnd demo\n";
}