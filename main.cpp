#include "NeuralNetwork.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat img = cv::imread("西屯藍天.jpg");
    cv::imshow("test img",img);
    cv::waitKey(0);

    cout << "Begin neural network regression demo" << endl;
    cout << "Goal is to predict the sin(x)" << endl;

    int numItems = 80;
    cout << "Programmatically generating " + to_string(numItems) + " training data items" << endl;

    //make 2D vector
    vector<double> row(2);
    vector<vector<double>> trainData(numItems, row);

    LKY::NeuralNetwork::Random rnd = LKY::NeuralNetwork::Random();

    //產生一個周期內的80個sin取樣點
    for (int i = 0; i < numItems; ++i)
    {
        double x = 6.4 * rnd.NextDouble(); // [0 to 2PI]
        double sx = sin(x);
        trainData[i][0] = x;
        trainData[i][1] = sx;
        printf("x=%lf, sx=%lf\n", x, sx);
    }
    cout << endl;
    cout << "Training data:" << endl;
    //Show.ShowMatrix(trainData, 3, 4, true);

    LKY::NeuralNetwork nn = LKY::NeuralNetwork(1, 12, 1, 0);

    int maxEpochs = 10000;
    double learnRate = 0.02;
    double momentum = 0.0005;
    nn.Train(trainData, maxEpochs, learnRate, momentum);

    vector<double> weights = nn.GetWeights();
    std::cout << "weights:" << endl;
    for(auto const &n : weights)
    {
        cout << ' ' << n;
    }

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