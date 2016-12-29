#include "Layers.hpp"

int main()
{
    //初始配置
    InputLayer*  inputLayer  = new InputLayer();
    HiddenLayer* hiddenLayer = new HiddenLayer();
    OutputLayer* outputLayer = new OutputLayer();

    inputLayer = new InputLayer(2,hiddenLayer);
    hiddenLayer = new HiddenLayer(2, (Layer*)inputLayer, (Layer*)outputLayer);
    outputLayer = new OutputLayer(2,hiddenLayer);

    hiddenLayer->InitializeWeights();
    outputLayer->InitializeWeights();

    //計算開始
    inputLayer->Input(vector<double>{2,2});
    hiddenLayer->ForwardPropagation();
    outputLayer->ForwardPropagation();
    vector<double> outputArray = outputLayer->GetOutput();

    //print
    for (double const output : outputArray)
    {
        printf("%lf, ",output);
    }cout << endl;
}