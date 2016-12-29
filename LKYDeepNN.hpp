#include "Layers.hpp"

class LKYDeepNN
{
    //各層指標
    private: InputLayer* inputLayer;
    private: vector<HiddenLayer*> hiddenLayerArray;
    private: OutputLayer* outputLayer = new OutputLayer();
    
    public: LKYDeepNN(int numInputNodes, vector<int> numHiddenNodes, int numOutputNodes)
    {
        //輸入層連結配置
        this->hiddenLayerArray = vector<HiddenLayer*>(numHiddenNodes.size()); //這行要先做, 不然沒東西傳入InputLayer建構子
        this->inputLayer = new InputLayer(numInputNodes, *(hiddenLayerArray.begin()));

        //隱藏層連結配置
        if(1 ==  this->hiddenLayerArray.size())
        {
            int numNode = numHiddenNodes.front();
            hiddenLayerArray.front() = new HiddenLayer(numNode, (Layer*)inputLayer, (Layer*)outputLayer);
        }
        else
        {
            for(vector<HiddenLayer*>::iterator it=hiddenLayerArray.begin(); it!=hiddenLayerArray.end(); it++)
            {
                //取得此層節點數
                int numNode = numHiddenNodes[it-hiddenLayerArray.begin()];

                //第一個隱藏層連結配置
                if(it==hiddenLayerArray.begin()){
                    *it = new HiddenLayer(numNode, (Layer*)inputLayer, (Layer*)*(it+1));
                    continue;}

                //最後一個隱藏層連結配置
                if(it!=hiddenLayerArray.end()-1){
                    *it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)outputLayer);
                    continue;}

                //中間隱藏層連結配置
                *it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)*(it+1));
            }
        }

        //輸出層連結配置
        outputLayer = new OutputLayer(numOutputNodes, hiddenLayerArray.back());

        //統一權重初始化
        this->InitializeWeights();
    }

    public: void InitializeWeights()
    {
        //權重初始化
        cout << "權重初始化" << endl;
        for (auto hiddenLayer : this->hiddenLayerArray)
        {
            hiddenLayer->InitializeWeights();
        }
        this->outputLayer->InitializeWeights();
    }

     public: vector<double> ForwardPropagation(vector<double> inputArray)
     {
        //輸入到輸入層節點
        this->inputLayer->Input(inputArray);
        
        //隱藏層順傳遞
        for (auto hiddenLayer : hiddenLayerArray)
        {
            hiddenLayer->ForwardPropagation();
        }
        
        //最後一個隱藏層到輸出層的順傳遞
        outputLayer->ForwardPropagation();

        //回傳輸出層輸出
        return outputLayer->GetOutput();
     }
};