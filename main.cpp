#include "Layers.hpp"

int main()
{
    //初始連結配置
    InputLayer*  inputLayer  = new InputLayer();
    vector<HiddenLayer*> hiddenLayerArray(6);// = new HiddenLayer();
    OutputLayer* outputLayer = new OutputLayer();

    inputLayer = new InputLayer(2, *(hiddenLayerArray.begin()));
    if(1 == hiddenLayerArray.size())
    {
        hiddenLayerArray.front() = new HiddenLayer(2, (Layer*)inputLayer, (Layer*)outputLayer);
    }
    else
    {
        for(vector<HiddenLayer*>::iterator it=hiddenLayerArray.begin(); it!=hiddenLayerArray.end(); it++)
        {
            if(it==hiddenLayerArray.begin()){
                *it = new HiddenLayer(2, (Layer*)inputLayer, (Layer*)*(it+1));continue;}

            if(it!=hiddenLayerArray.end()-1){
                *it = new HiddenLayer(2, (Layer*)*(it-1), (Layer*)outputLayer);continue;}

            *it = new HiddenLayer(2, (Layer*)*(it-1), (Layer*)*(it+1));
        }
    }
    outputLayer = new OutputLayer(2,hiddenLayerArray.back());
    
    //權重初始化
    cout << "權重初始化" << endl;
    for (auto hiddenLayer : hiddenLayerArray)
    {
        hiddenLayer->InitializeWeights();
    }
    outputLayer->InitializeWeights();

    //計算開始
    cout << "計算開始" << endl;
    inputLayer->Input(vector<double>{2,2});
    for (auto hiddenLayer : hiddenLayerArray)
    {
        hiddenLayer->ForwardPropagation();
    }
    outputLayer->ForwardPropagation();
    vector<double> outputArray = outputLayer->GetOutput();

    //print
    for (double const output : outputArray)
    {
        printf("%lf, ",output);
    }cout << endl;
}