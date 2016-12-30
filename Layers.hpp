#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include "Activation.hpp"
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

    //活化函數
    private: Activation* activation = NULL;
    public: void SetActivation(Activation* activation)
    {
        this->activation = activation;
    }

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

    public: void ForwardPropagation()
    {
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
        {
            this->nodes = this->activation->Forward(this->nodes);
        }
    }

    public: void BackPropagation(double learningRate)
    {
        vector<vector<double>> deltaWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
        vector<double> deltaGradients(this->hiddenBiases.size());

        // if(desiredOutValues.size() != this->nodes.size())
        // {
        //     cout << "ERROR: desiredOutValues.size() != this->nodes.size()" << endl;
        //     exit(EXIT_FAILURE);
        // }

        // //權重變化計算與更新
        // for (size_t j = 0; j < deltaWeights.size(); ++j)
        // {
        //     if(deltaWeights[j].size() != this->intoWeights[j].size())
        //     {//長度檢查
        //         cout << "ERROR: deltaWeights[j].size() != this->intoWeights[j]" << endl;
        //         exit(EXIT_FAILURE);
        //     }

        //     for (size_t i = 0; i < deltaWeights[j].size(); ++i)
        //     {
        //         //計算輸出與目標的差值
        //         //∆w kj = ε (t k − a k )a k (1 − a k ) a j
        //         double err = desiredOutValues[i] - this->nodes[i];
        //         deltaWeights[j][i] = err*this->activation->Derivative(this->nodes[i])*this->previousLayer->nodes[j];
                
        //         //更新權重
        //         this->intoWeights[j][i] += learningRate*deltaWeights[j][i];
        //     }
        // }

        // //基底變化計算與更新
        // for (size_t i = 0; i < deltaGradients.size(); ++i)
        // {
        //     double err = desiredOutValues[i] - this->nodes[i];
        //     deltaGradients[i] = err*this->activation->Derivative(this->nodes[i]);
        //     this->outBiases[i] += deltaGradients[i]; 
        // }
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

    //活化函數
    private: Activation* activation = NULL;
    public: void SetActivation(Activation* activation)
    {
        this->activation = activation;
    }

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

        cout << "completed output Layer InitializeWeights()" << endl;
    }

    public: void ForwardPropagation()
    {
        //節點的乘積與和
        for (size_t j = 0; j < this->nodes.size(); ++j) // compute i-h sum of weights * inputNodes
        {
            for (size_t i = 0; i < this->previousLayer->nodes.size(); ++i)
            {
                this->nodes[j] += this->previousLayer->nodes[i] * this->intoWeights[i][j]; // note +=
            }

            this->nodes[j] += this->outBiases[j];
        }

        //活化函數
        if(NULL == this->activation)
        {
            cout << "ERROR: 忘記配置活化函數" << endl;
            exit(EXIT_FAILURE);
        }
        else
        {
            this->nodes = this->activation->Forward(this->nodes);
        }
    }

    public: void BackPropagation(double learningRate, vector<double> desiredOutValues)
    {
        vector<vector<double>> deltaWeights = MakeMatrix(this->previousLayer->nodes.size(), this->nodes.size(), 1.0);
        vector<double> deltaGradients(this->outBiases.size());

        if(desiredOutValues.size() != this->nodes.size())
        {
            cout << "ERROR: desiredOutValues.size() != this->nodes.size()" << endl;
            exit(EXIT_FAILURE);
        }

        //權重變化計算與更新
        for (size_t j = 0; j < deltaWeights.size(); ++j)
        {
            if(deltaWeights[j].size() != this->intoWeights[j].size())
            {//長度檢查
                cout << "ERROR: deltaWeights[j].size() != this->intoWeights[j]" << endl;
                exit(EXIT_FAILURE);
            }

            for (size_t i = 0; i < deltaWeights[j].size(); ++i)
            {
                //計算輸出與目標的差值
                //∆w kj = ε (t k − a k )a k (1 − a k ) a j
                double err = desiredOutValues[i] - this->nodes[i];
                double activationOutput = this->activation->Derivative(this->nodes[i]);
                printf("err = %lf, activationOutput=%lf, previousLayer->nodes[j]=%lf\n" ,err , activationOutput, previousLayer->nodes[j]);
                deltaWeights[j][i] = err*activationOutput*this->previousLayer->nodes[j];
                
                //更新權重
                printf("update before: %lf, ",this->intoWeights[j][i]);
                this->intoWeights[j][i] += learningRate*deltaWeights[j][i];
                printf("update after:  %lf\n",this->intoWeights[j][i]);

            }cout << endl;
        }

        //基底變化計算與更新
        for (size_t i = 0; i < deltaGradients.size(); ++i)
        {
            double err = desiredOutValues[i] - this->nodes[i];
            deltaGradients[i] = err*this->activation->Derivative(this->nodes[i]);
            this->outBiases[i] += deltaGradients[i]; 
        }
    }

    public: vector<double> GetOutput()
    {
        return this->nodes;
    }
};