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
        //===================== step 1: 各層實體配置 ===================== 
        this->inputLayer = new InputLayer();
        this->hiddenLayerArray = vector<HiddenLayer*>(numHiddenNodes.size()); //這行要先做, 不然沒東西傳入InputLayer建構子
        for(auto& aHiddenLayer : this->hiddenLayerArray)
        {
            aHiddenLayer = new HiddenLayer();
        }
        printf("最後一個隱藏層位址=%p\n",hiddenLayerArray.back());
        this->outputLayer = new OutputLayer();


        //===================== step 1: 各層連結配置 & 節點初始化 =====================
        //noteic: 這一層不能再做實體配置，不然會改變各層的位址，先前建立好的link會壞掉
        // 輸入層
        this->inputLayer->SetNextLayer(hiddenLayerArray.front());
        (this->inputLayer)->SetNode(numInputNodes);

        //隱藏層
        if(1 ==  this->hiddenLayerArray.size())
        {
            int numNode = numHiddenNodes.front();
            this->hiddenLayerArray.front()->SetPrevLayer((Layer*)inputLayer);
            this->hiddenLayerArray.front()->SetNextLayer((Layer*)outputLayer);
            this->hiddenLayerArray.front()->SetNode(numNode);
            this->hiddenLayerArray.front()->SetActivation(new Tanh());
        }
        else
        {
            for(vector<HiddenLayer*>::iterator it=hiddenLayerArray.begin(); it!=hiddenLayerArray.end(); it++)
            {
                //取得此層節點數
                int numNode = numHiddenNodes[it-hiddenLayerArray.begin()];
                
                if(it==hiddenLayerArray.begin())
                {//第一個隱藏層連結配置
                    this->hiddenLayerArray.front()->SetPrevLayer((Layer*)inputLayer);
                    this->hiddenLayerArray.front()->SetNextLayer((Layer*)*(it+1));
                    //*it = new HiddenLayer(numNode, (Layer*)inputLayer, (Layer*)*(it+1));
                }
                else if(it==hiddenLayerArray.end()-1)
                {//最後一個隱藏層連結配置
                    this->hiddenLayerArray.back()->SetPrevLayer((Layer*)*(it-1));
                    this->hiddenLayerArray.back()->SetNextLayer((Layer*)outputLayer);
                    //*it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)outputLayer);
                    printf("最後一個隱藏層位址=%p\n",*it);
                }
                else
                {//中間隱藏層連結配置
                    (*it)->SetPrevLayer((Layer*)*(it-1));
                    (*it)->SetNextLayer((Layer*)*(it+1));
                    //*it = new HiddenLayer(numNode, (Layer*)*(it-1), (Layer*)*(it+1));
                    cout << "中間隱藏層連結配置" << endl;
                }

                //節點 & 活化函數配置
                (*it)->SetNode(numNode);
                (*it)->SetActivation(new Tanh());
            }
        }

        //輸出層連結 & 活化函數配置
        this->outputLayer->SetPrevLayer(hiddenLayerArray.back());
        this->outputLayer->SetNode(numOutputNodes);
        this->outputLayer->SetActivation(new Tanh());
        //outputLayer = new OutputLayer(numOutputNodes, hiddenLayerArray.back());
        
        //最後一個隱藏層重新配置(這邊方法有點爛)
        // int numNode = numHiddenNodes.back();//取得最後一個隱藏層應有節點數
        // hiddenLayerArray.back()->SetPrevLayer((Layer*)*(hiddenLayerArray.end()-2));
        // hiddenLayerArray.back()->SetNextLayer((Layer*)outputLayer);
        printf("最後一個隱藏層位址=%p\n",hiddenLayerArray.back());
        // hiddenLayerArray.back()->SetActivation(new Tanh());

        // //倒數第二個隱藏層重新配置(這邊方法有點爛)
        // numNode = numHiddenNodes.back();//取得最後一個隱藏層應有節點數
        // hiddenLayerArray[hiddenLayerArray.size()-2] = new HiddenLayer(numNode, (Layer*)*(hiddenLayerArray.end()-3), (Layer*)*(hiddenLayerArray.end()-1));
        // hiddenLayerArray[hiddenLayerArray.size()-2]->SetActivation(new Tanh());
        
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
        cout << "輸入資料到輸入層節點" << endl;
        this->inputLayer->Input(inputArray);
        
        //隱藏層順傳遞
        cout << "隱藏層順傳遞" << endl;
        for (auto hiddenLayer : hiddenLayerArray)
        {
            hiddenLayer->ForwardPropagation();
        }
        
        //最後一個隱藏層到輸出層的順傳遞
        cout << "最後一個隱藏層到輸出層的順傳遞" << endl;
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