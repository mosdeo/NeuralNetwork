#include "Layer.hpp"
#include "InputLayer.hpp"
#include "HiddenLayer.hpp"
#include "OutputLayer.hpp"
using namespace std;

void HiddenLayer::SetActivation(Activation* activation)
{
    this->activation = activation;
}

HiddenLayer::HiddenLayer(){};
HiddenLayer::HiddenLayer(int numNodes, Layer* previousLayer, Layer* nextLayer)
{
    // bool isInputLayer = typeid(nextLayer)==typeid(InputLayer*);
    // bool isHiddenLayer = typeid(nextLayer)==typeid(HiddenLayer*);
    // bool isOutputLayer = typeid(nextLayer)==typeid(OutputLayer*);
    // printf("%d,%d,%d\n",isInputLayer, isHiddenLayer, isOutputLayer);
    // if(!isHiddenLayer || !isOutputLayer)
    // {
    //     cout << "HiddenLayer 的下一層必須是 HiddenLayer 或 OutputLayer" << endl;
    //     exit(EXIT_FAILURE);
    // }

    // isInputLayer = typeid(previousLayer)==typeid(InputLayer*);
    // isHiddenLayer = typeid(previousLayer)==typeid(HiddenLayer*);
    // isOutputLayer = typeid(previousLayer)==typeid(OutputLayer*);
    // if(isHiddenLayer && isOutputLayer)
    // {
    //     cout << "HiddenLayer 的上一層必須是 HiddenLayer 或 InputLayer" << endl;
    //     exit(EXIT_FAILURE);
    // }


    this->previousLayer = previousLayer;
    this->nextLayer = nextLayer;

    this->nodes  = vector<double>(numNodes);
    this->hiddenBiases = vector<double>(numNodes,0); //numNodes double with value 0

    this->wGrads = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 0.0);
    this->oGrads = vector<double>(this->hiddenBiases.size());
}

void HiddenLayer::InitializeWeights()
{   
    this->intoWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);

    const double hi = 1/(sqrt(this->nodes.size()));
    const double lo = -hi;

    //std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni_noise(lo, hi); // guaranteed unbiased

    for (size_t j = 0; j < this->intoWeights.size(); ++j)
        for (size_t i = 0; i < this->intoWeights[j].size(); ++i)
    {
        this->intoWeights[j][i] = uni_noise(rng);
    }

    cout << "completed hidden Layer InitializeWeights()" << endl;
}

void HiddenLayer::ForwardPropagation()
{
    //將自己的節點歸零，因為要存放上一級傳來的運算結果，不能累積。
    this->nodes = vector<double>(this->nodes.size() ,0.0);

    //節點的乘積與和
    for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
    {
        for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
        {
            this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=
        }

        this->nodes[j] += this->hiddenBiases[j];
    }

    //活化函數
    if(NULL == this->activation)
    {
        cout << "ERROR: 忘記配置活化函數" << endl;
        exit(EXIT_FAILURE);
    }
    else
    {//將自身節點全部跑一次活化函數
        this->nodes = this->activation->Forward(this->nodes);
    }
}

void HiddenLayer::BackPropagation(double learningRate)
{
    for(size_t j=0 ; j < this->wGrads.size() ; j++)
    {
        for(size_t i=0 ; i < this->wGrads[j].size() ; i++)
        {
            if(typeid(*(this->nextLayer)) == typeid(InputLayer))
            {
                cout << "ERROR: HiddenLayer 的下一層不能是 InputLayer." << endl;
                exit(EXIT_FAILURE);
            }

            if(typeid(*(this->nextLayer)) == typeid(OutputLayer))
            {
                cout << "ERROR: HiddenLayer 的下一層不能是 OutputLayer." << endl;
                exit(EXIT_FAILURE);
            }

            cout << "mark" << endl;
            cout << this->previousLayer->ToString() << endl;
            cout << this->nextLayer->ToString() << endl;
            

            double pervGrad = this->nextLayer->wGrads[j][i];cout << "mark" << endl;
            double derivativeActivation = this->activation->Derivative(this->nodes[i]);
            double pervInput = this->previousLayer->nodes[j];
            this->wGrads[j][i] = pervGrad*derivativeActivation*pervInput;

            //更新權重
            this->intoWeights[j][i] -= learningRate*this->wGrads[j][i];
        }
    }
}

vector<double> HiddenLayer::GetOutput()
{
    return this->nodes;
}