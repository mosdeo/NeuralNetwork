#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"

class LKYDeepNN
{
    //各層指標
    private: InputLayer* inputLayer;
    private: vector<HiddenLayer*> hiddenLayerArray;
    private: OutputLayer* outputLayer;
    
    public: LKYDeepNN(int numInputNodes, vector<int> numHiddenNodes, int numOutputNodes)
    {
        //輸入層連結配置
        this->hiddenLayerArray = vector<HiddenLayer*>(numHiddenNodes.size()); //這行要先做, 不然沒東西傳入InputLayer建構子
        //this->inputLayer = new InputLayer(numInputNodes, *(hiddenLayerArray.begin()));
        this->inputLayer = new InputLayer(numInputNodes, hiddenLayerArray.front());

        //隱藏層連結配置
        if(1 ==  this->hiddenLayerArray.size())
        {
            int numNode = numHiddenNodes.front();
            hiddenLayerArray.front() = new HiddenLayer(numNode, (Layer*)inputLayer, (Layer*)outputLayer);
            hiddenLayerArray.front()->SetActivation(new Tanh());
        }
        else
        {
            for(vector<HiddenLayer*>::iterator it=hiddenLayerArray.begin(); it!=hiddenLayerArray.end(); it++)
            {
                //取得此層節點數
                int numNode = numHiddenNodes[it-hiddenLayerArray.begin()];
                
                if(it==hiddenLayerArray.begin())
                {//第一個隱藏層連結配置
                    *it = new HiddenLayer(numNode, (Layer*)inputLayer, (Layer*)*(it+1));
                }
                else if(it==hiddenLayerArray.end()-1)
                {//最後一個隱藏層連結配置
                    *it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)outputLayer);
                }
                else
                {//中間隱藏層連結配置
                    *it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)*(it+1));
                    cout << "中間隱藏層連結配置" << endl;
                }

                //活化函數配置
                (*it)->SetActivation(new Tanh());
            }
        }

        //輸出層連結配置
        outputLayer = new OutputLayer(numOutputNodes, hiddenLayerArray.back());
        
        //最後一個隱藏層重新配置(這邊方法有點爛)
        //int numNode = numHiddenNodes[numHiddenNodes.size()-1];
        int numNode = numHiddenNodes.back();//取得最後一個隱藏層應有節點數
        hiddenLayerArray.back() = new HiddenLayer(numNode, (Layer*)*(hiddenLayerArray.end()-2), (Layer*)outputLayer);
        hiddenLayerArray.back()->SetActivation(new Tanh());
        
        //輸出層活化函數配置
        outputLayer->SetActivation(new Tanh());

        //統一權重初始化
        this->InitializeWeights();

        cout << "測試最後一個隱藏層連結配置" << endl;
        cout << hiddenLayerArray.back()->ToString() << endl;
        cout << hiddenLayerArray.back()->nextLayer->ToString() << endl;
        cout << "======================" << endl;
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
        //輸入資料到輸入層節點
        //cout << "輸入資料到輸入層節點" << endl;
        this->inputLayer->Input(inputArray);
        
        //隱藏層順傳遞
        //cout << "隱藏層順傳遞" << endl;
        for (auto hiddenLayer : hiddenLayerArray)
        {
            hiddenLayer->ForwardPropagation();
        }
        
        //最後一個隱藏層到輸出層的順傳遞
        //cout << "最後一個隱藏層到輸出層的順傳遞" << endl;
        this->outputLayer->ForwardPropagation();

        //回傳輸出層輸出
        return this->outputLayer->GetOutput();
     }

     //public: void Training(vector<vector<double>> trainData, int totalEpochs, double learnRate, double momentum)
     public: void Training(double learningRate, vector<double> desiredOutValues)
     {
         this->outputLayer->BackPropagation(learningRate, desiredOutValues);

         for(vector<HiddenLayer*>::reverse_iterator r_it=hiddenLayerArray.rbegin(); r_it!=hiddenLayerArray.rend(); r_it++)
         {
            (*r_it)->BackPropagation(learningRate);
         }
     }
};