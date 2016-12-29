#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

class Layer
{
    friend class HiddenLayer;

    //層節點
    protected: vector<double> nodes;

    protected: vector<vector<double>> MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
    {
        vector<double> row;
        row.assign(cols, v); //配置一個row的大小
        vector<vector<double>> array_2D;
        array_2D.assign(rows, row); //配置2維

        return array_2D;
    }
};

class HiddenLayer;

class InputLayer: private Layer
{
    //後層
    protected: HiddenLayer* nextLayer;

    public: InputLayer(){};
    public: InputLayer(int numNodes, HiddenLayer* nextLayer)
    {
        this->nextLayer = nextLayer;
        this->nodes  = vector<double>(numNodes);
    }

    public: void Input(const vector<double> inputArray)
    {
        if(this->nodes.size() <= inputArray.size())
        {
            this->nodes = inputArray;
        }
        else
        {//長度檢查未通過
            cout << "ERROR: this->nodes.size() > inputArray.size()" << endl;
            exit(EXIT_FAILURE);
        }
    }
};

class HiddenLayer: private Layer
{
    friend class OutputLayer;

    //順向進入的權重與基底
    protected: vector<vector<double>> intoWeights;
    protected: vector<double> hiddenBiases;
    
    //前後層
    protected: Layer* previousLayer;
    protected: Layer* nextLayer;

    public: HiddenLayer(){};
    public: HiddenLayer(int numNodes, Layer* previousLayer, Layer* nextLayer)
    {
        this->previousLayer = previousLayer;
        this->nextLayer = nextLayer;

        this->nodes  = vector<double>(numNodes);
        this->hiddenBiases = vector<double>(numNodes,0); //numNodes double with value 0
    }

    public: void InitializeWeights()
    {
        this->intoWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
    }

    private: vector<double> Activation(vector<double> nodeSum)
    {
        vector<double> result(nodeSum.size());

        for (size_t i = 0; i < nodeSum.size(); ++i)
            result[i] = nodeSum[i];//暫時先這樣

        return result;
    }

    public: void ForwardPropagation()
    {
        //節點的乘積與和
        for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
            for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
                this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=

        //活化函數
        this->nodes = Activation(this->nodes);
    }

    public: void BackPropagation()
    {
        vector<vector<double>> grads = MakeMatrix(this->nodes.size(), this->nextLayer->nodes.size(), 1.0); // hidden-to-output weights gradients
        vector<double> biasesGrads(this->nodes.size()); // output biases gradients
    }

    public: vector<double> GetOutput()
    {
        return this->nodes;
    }
};

class OutputLayer: private Layer
{
    //順向進入的權重與基底
    protected: vector<vector<double>> intoWeights;
    protected: vector<double> outBiases;

    //前層
    protected: HiddenLayer* previousLayer;

    public: OutputLayer(){};
    public: OutputLayer(int numNodes, HiddenLayer* previousLayer)
    {
        this->previousLayer = previousLayer;

        this->nodes  = vector<double>(numNodes);
        this->outBiases = vector<double>(numNodes,0); //numNodes double with value 0
        
    }

    public: void InitializeWeights()
    {
        this->intoWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
    }

    private: vector<double> Activation(vector<double> nodeSum)
    {
        vector<double> result(nodeSum.size());

        for (size_t i = 0; i < nodeSum.size(); ++i)
            result[i] = nodeSum[i];//暫時先這樣

        return result;
    }

    public: void ForwardPropagation()
    {
        //節點的乘積與和
        for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
            for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
                this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=

        //活化函數
        this->nodes = Activation(this->nodes);
    }

    public: vector<double> GetOutput()
    {
        return this->nodes;
    }
};