#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <cstdlib>
#include <iostream>
#include <algorithm> // std::random_shuffle
#include <thread>
#include <cmath>
#include <vector>
using namespace std;

namespace LKY
{
class NeuralNetwork
{
    //各層節點數目
    private: int numInput; // number input nodes
    private: int numHidden;
    private: int numOutput;
    
    //層到層之間的權重數目
    private: int num_In_to_Hidden; //in-hidden
    private: int num_Hidden_to_Hidden; //hidden-hidden
    private: int num_Hidden_to_Out; //hidden-out
    
    //總權重數目
    private: size_t numWeights;

    //各層節點
    private: vector<double> inputNodes;  //輸入層
    private: vector<double> hiddenNodes1;//第1隱藏層
    private: vector<double> hiddenNodes2;//第2隱藏層
    private: vector<double> outputNodes; //輸出層
    
    //各層權重
    private: vector<vector<double>> ihWeights; // input-hidden
    private: vector<double> h1Biases;
    private: vector<vector<double>> hhWeights; // hidden-hidden
    private: vector<double> h2Biases;
    private: vector<vector<double>> hoWeights; // hidden-output
    private: vector<double> oBiases;

    //倒傳遞(back-prop)相關變數
    private: vector<vector<double>> ihPrevWeightsDelta;//input-hidden權重變化
    private: vector<double> h1PrevBiasesDelta;         //第1隱藏層截距變化
    private: vector<vector<double>> hhPrevWeightsDelta;//hidden-hidden權重變化
    private: vector<double> h2PrevBiasesDelta;         //第2隱藏層截距變化
    private: vector<vector<double>> hoPrevWeightsDelta;//hidden-output權重變化
    private: vector<double> oPrevBiasesDelta;          //輸出層截距變化

    public: bool isVisualizeTraining = false;
    private: bool isClassification = false;
    public: void SetClassification()
    {//若輸出層node少於2則不可為分類器，只能做回歸用
        if(2 > this->numOutput)
        {
            printf("Can't as Classifier, 2 > numOutput == %d.",this->numOutput);
            exit(EXIT_FAILURE);
        }
        else
        {
            this->isClassification = true ;
        }
    }

    private: vector<double> trainError;
    public: vector<double> GetTrainError(){return this->trainError;}

    public: ~NeuralNetwork()
    {
        this->inputNodes.clear();
        this->hiddenNodes1.clear();
        this->hiddenNodes2.clear();
        this->outputNodes.clear();

        this->ihWeights.clear();
        this->h1Biases.clear();
        this->hhWeights.clear();
        this->h2Biases.clear();
        this->hoWeights.clear();
        this->oBiases.clear();
    }

    public: class Random
    {
        private: std::mt19937_64 randGen;
        public: Random()
        {
            randGen = std::mt19937_64(time(NULL));
        }

        public: Random(unsigned int seed)
        {
            randGen = std::mt19937_64(seed);
        }

        public: int Next(int minValue, int maxValue)
        {
            return this->NextDouble() * (maxValue - minValue) + minValue;
        }

        public: double NextDouble()
        {
            
            return ((double)randGen()) / randGen.max();
        }
    };

    private: Random rnd;

    public: NeuralNetwork(int numInput, int numHidden, int numOutput, int seed)
    {
        this->numInput = numInput;
        this->numHidden = numHidden;
        this->numOutput = numOutput;

        this->num_In_to_Hidden = (numInput * numHidden) + numHidden; //in-hidden
        this->num_Hidden_to_Hidden = (numHidden * numHidden) + numHidden; //hidden-hidden
        this->num_Hidden_to_Out = (numHidden * numOutput) + numOutput; //hidden-out
        this->numWeights = this->num_In_to_Hidden + this->num_Hidden_to_Hidden + this->num_Hidden_to_Out;

        this->inputNodes.resize(numInput);
        this->hiddenNodes1.resize(numHidden);
        this->hiddenNodes2.resize(numHidden);
        this->outputNodes.resize(numOutput);

        this->ihWeights = MakeMatrix(numInput, numHidden, 0.0);
        this->h1Biases.resize(numHidden);

        this->hhWeights = MakeMatrix(numHidden, numHidden, 0.0);
        this->h2Biases.resize(numHidden);

        this->hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
        this->oBiases.resize(numOutput);

        // back-prop momentum specific arrays
        ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
        h1PrevBiasesDelta = vector<double>(numHidden);
        hhPrevWeightsDelta = MakeMatrix(numHidden, numHidden, 0.0);
        h2PrevBiasesDelta = vector<double>(numHidden);
        hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
        oPrevBiasesDelta = vector<double>(numOutput);

        this->rnd = Random(seed);
        this->InitializeWeights();
    }                              // ctor

    private: static vector<vector<double>> MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
    {
        vector<double> row;
        row.assign(cols, v); //配置一個row的大小
        vector<vector<double>> array_2D;
        array_2D.assign(rows, row); //配置2維

        return array_2D;
    }

    public: void InitializeWeights() // helper for ctor
    {
        vector<double> initialWeights(numWeights);
        double hi , lo;

        //in-hidden weight Set 
        hi = 1/(sqrt(numInput));
        lo = -hi;
        for (int i = 0; i < num_In_to_Hidden; ++i)
        {
            initialWeights[i] = (hi-lo) *  rnd.NextDouble() + lo;
        }

        //hidden-hidden weight Set
        hi = 1/(sqrt(numHidden));
        lo = -hi;
        for (int i = num_In_to_Hidden; i < num_In_to_Hidden+num_Hidden_to_Hidden; ++i)
        {
            initialWeights[i] = (hi-lo) *  rnd.NextDouble() + lo;
        }

        //hidden-out weight Set
        hi = 1/(sqrt(numHidden));
        lo = -hi;
        for (size_t i = num_In_to_Hidden+num_Hidden_to_Hidden; i < numWeights; ++i)
        {
            initialWeights[i] = (hi-lo) *  rnd.NextDouble() + lo;
        }

        this->SetWeights(initialWeights);
    }

    public: void SetWeights(vector<double> weights)
    {
        // copy serialized weights and biases in weights[] array
        // to i-h weights, i-h biases, h-o weights, h-o biases

        if (weights.size() != numWeights)
        {
            cout << "throw new exception(\"Bad weights array in SetWeights\");" << endl;
            exit(EXIT_FAILURE);
        }

        int w = 0; // points into weights param

        //in-hidden weight Set
        for (size_t i = 0; i < ihWeights.size(); ++i)
            for (size_t j = 0; j < ihWeights[i].size(); ++j)
                ihWeights[i][j] = weights[w++];

        for (size_t j = 0; j < h1Biases.size(); ++j)
            h1Biases[j] = weights[w++];

        //hidden-hidden weight Set
        for (size_t j = 0; j < hhWeights.size(); ++j)
            for (size_t k = 0; k < hhWeights[j].size(); ++k)
                hhWeights[j][k] = weights[w++];

        for (size_t k = 0; k < h2Biases.size(); ++k)
            h2Biases[k] = weights[w++];

        //hidden-out weight Set
        for (size_t j = 0; j < hoWeights.size(); ++j)
            for (size_t k = 0; k < hoWeights[j].size(); ++k)
                hoWeights[j][k] = weights[w++];

        for (size_t k = 0; k < oBiases.size(); ++k)
            //oBiases[k] = weights[k++];
            oBiases[k] = weights[w++];
    }

    public: vector<double> GetWeights()
    {
        vector<double> result(numWeights);
        int w = 0;

        //in-hidden weight Set
        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                result[w++] = ihWeights[i][j];

        for (int j = 0; j < numHidden; ++j)
            result[w++] = h1Biases[j];

        //hidden-hidden weight Set
        for (int i = 0; i < numHidden; ++i)
            for (int j = 0; j < numHidden; ++j)
                result[w++] = hhWeights[i][j];

        for (int j = 0; j < numHidden; ++j)
            result[w++] = h2Biases[j];

        //hidden-out weight Set
        for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
                result[w++] = hoWeights[j][k];

        for (int k = 0; k < numOutput; ++k)
            result[w++] = oBiases[k];

        return result;
    }

    public: vector<double> ComputeOutputs(vector<double> xValues)
    {
        vector<double> h1Sums(numHidden); // hidden nodes sums scratch array
        vector<double> h2Sums(numHidden);// hidden nodes sums scratch array
        vector<double> oSums(numOutput); // output nodes sums

        for (int i = 0; i < numInput; ++i) // copy x-values to inputNodes
            this->inputNodes[i] = xValues[i];

        //First hidden node compute
        for (int j = 0; j < numHidden; ++j) // compute i-h sum of weights * inputNodes
            for (int i = 0; i < numInput; ++i)
                h1Sums[j] += this->inputNodes[i] * this->ihWeights[i][j]; // note +=

        for (int j = 0; j < numHidden; ++j) // add biases to input-to-hidden sums
            h1Sums[j] += this->h1Biases[j];

        for (int j = 0; j < numHidden; ++j)        // apply activation
            this->hiddenNodes1[j] = HyperTan(h1Sums[j]); // hard-coded


        //Second hidden node compute
        for (int j = 0; j < numHidden; ++j) // compute i-h sum of weights * inputNodes
            for (int i = 0; i < numHidden; ++i)
                h2Sums[j] += this->hiddenNodes1[j] * this->hhWeights[i][j]; // note +=

        for (int j = 0; j < numHidden; ++j) // add biases to input-to-hidden sums
            h2Sums[j] += this->h2Biases[j];

        for (int j = 0; j < numHidden; ++j)        // apply activation
            this->hiddenNodes2[j] = HyperTan(h2Sums[j]); // hard-coded


        //Out node compute
        for (int k = 0; k < numOutput; ++k) // compute h-o sum of weights * hOutputs
            for (int j = 0; j < numHidden; ++j)
                oSums[k] += this->hiddenNodes2[j] * hoWeights[j][k];

        for (int k = 0; k < numOutput; ++k) // add biases to input-to-hidden sums
            oSums[k] += oBiases[k];

        std::copy(oSums.begin(), oSums.begin() + this->numOutput, this->outputNodes.begin()); // copy without activation

        vector<double> retResult(numOutput); // could define a GetOutputs
        std::copy(this->outputNodes.begin(), this->outputNodes.begin() + this->numOutput, retResult.begin());

        //根據演算法的不同，決定輸出層是否要疊上softmax
        return (this->isClassification ? this->Softmax(retResult) : retResult );
    }

    private: static double HyperTan(double x)
    {
        if (x < -20.0)
            return -1.0; // approximation is correct to 30 decimals
        else if (x > 20.0)
            return 1.0;
        else
            return tanh(x);
    }

    private: static vector<double> Softmax(vector<double> oSums)
    {
        // does all output nodes at once so scale
        // doesn't have to be re-computed each time

        // if (oSums.Length < 2) throw . . .
        vector<double> result(oSums.size());

        double sum = 0.0;
        for (size_t i = 0; i < oSums.size(); ++i)
            sum += exp(oSums[i]);

        for (size_t i = 0; i < oSums.size(); ++i)
            result[i] = exp(oSums[i]) / sum;

        return result; // now scaled so that xi sum to 1.0
    }

    public: void (*eventInTraining)(LKY::NeuralNetwork,int,int, const vector<vector<double>>& displayData) = NULL;
    public: void Train(vector<vector<double>> trainData, int maxEpochs, double learnRate, double momentum)
    {
        // train using back-prop
        // back-prop specific arrays
        vector<vector<double>> hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weights gradients
        vector<double> obGrads(numOutput);                                      // output biases gradients

        vector<vector<double>> hhGrads = MakeMatrix(numHidden, numHidden, 0.0); // hidden-to-hidden weights gradients
        vector<double> hhbGrads(numHidden);                                      // hidden 2 biases gradients

        vector<vector<double>> ihGrads = MakeMatrix(numInput, numHidden, 0.0); // input-to-hidden weights gradients
        vector<double> hbGrads(numHidden);                                     // hidden biases gradients

        vector<double> oSignals(numOutput); // signals == gradients w/o associated input terms
        vector<double> h1Signals(numHidden); // hidden1 node signals
        vector<double> h2Signals(numHidden); // hidden2 node signals
        

        // train a back-prop style NN regression using learning rate and momentum
        int epoch = 0;
        vector<double> xValues(numInput);  // inputNodes
        vector<double> tValues(numOutput); // target values

        vector<int> sequence(trainData.size());
        for (size_t i = 0; i < trainData.size(); ++i)
            sequence[i] = i;

        while (epoch < maxEpochs)
        {
            ++epoch; // immediately to prevent display when 0
            this->trainError.push_back(this->Error(trainData)); //計算當下訓練誤差,並存入

            if(NULL != this->eventInTraining) //繪製訓練過程testData
            {//呼叫事件
                //this->eventInTraining(*this,maxEpochs,epoch);
                thread th(this->eventInTraining, *this, maxEpochs, epoch, trainData);
                th.join();
            }

            Shuffle(sequence); // visit each training data in random order
            for (size_t ii = 0; ii < trainData.size(); ++ii)
            {
                int idx = sequence[ii];

                std::copy(trainData[idx].begin(), trainData[idx].begin() + numInput, xValues.begin());
                std::copy(trainData[idx].begin() + numInput, trainData[idx].begin() + numInput + numOutput, tValues.begin());

                ComputeOutputs(xValues); // copy xValues in, compute outputs

                // indices: i = inputNodes, j = hiddens, k = outputs

                // 1. compute output nodes signals (assumes constant activation)
                for (int k = 0; k < numOutput; ++k)
                {//計算輸出與目標的差值
                    double derivative = 1.0; // for dummy output activation f'
                    oSignals[k] = (tValues[k] - outputNodes[k]) * derivative;
                }

                // 2. compute hidden-to-output weights gradients using output signals
                for (int j = 0; j < numHidden; ++j)
                    for (int k = 0; k < numOutput; ++k)
                        hoGrads[j][k] = oSignals[k] * hiddenNodes2[j];

                // 2b. compute output biases gradients using output signals
                for (int k = 0; k < numOutput; ++k)
                    obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                // 3. compute hidden nodes signals
                for (int j = 0; j < numHidden; ++j)
                {
                    double sum = 0.0; // need sums of output signals times hidden-to-output weights
                    for (int k = 0; k < numOutput; ++k)
                    {
                        sum += oSignals[k] * hoWeights[j][k];
                    }
                    double derivative = (1 + hiddenNodes2[j]) * (1 - hiddenNodes2[j]); // for tanh
                    //double derivative = 1 - pow(hiddenNodes2[j],2); // for tanh
                    h2Signals[j] = sum * derivative;
                }

                //======== Hidden-Hidden Layer =============
                // 2. compute hidden-to-hidden weights gradients
                for (int j = 0; j < numHidden; ++j)
                    for (int k = 0; k < numHidden; ++k)
                        hhGrads[j][k] = h2Signals[k] * hiddenNodes1[j];

                // 2b. compute hidden biases gradients
                for (int k = 0; k < numHidden; ++k)
                    hhbGrads[k] = h2Signals[k] * 1.0;

                // 3. compute hidden nodes signals
                for (int j = 0; j < numHidden; ++j)
                {
                    double sum = 0.0; // need sums of hidden signals times hidden-to-hidden weights
                    for (int k = 0; k < numHidden; ++k)
                    {
                        sum += h2Signals[k] * hhWeights[j][k];
                    }
                    double derivative = (1 + hiddenNodes1[j]) * (1 - hiddenNodes1[j]); // for tanh
                    //double derivative = 1 - pow(hiddenNodes1[j],2); // for tanh
                    h1Signals[j] = sum * derivative;
                }


                // 4. compute input-hidden weights gradients
                for (int i = 0; i < numInput; ++i)
                    for (int j = 0; j < numHidden; ++j)
                        ihGrads[i][j] = h1Signals[j] * inputNodes[i];

                // 4b. compute hidden node biases gradienys
                for (int j = 0; j < numHidden; ++j)
                    hbGrads[j] = h1Signals[j] * 1.0; // dummy 1.0 input

                // == update weights and biases

                // 1. update input-to-hidden weights
                for (int i = 0; i < numInput; ++i)
                {
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = ihGrads[i][j] * learnRate;
                        ihWeights[i][j] += delta;
                        ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                        ihPrevWeightsDelta[i][j] = delta; // save for next time
                    }
                }

                // 2. update hidden biases
                for (int j = 0; j < numHidden; ++j)
                {
                    double delta = hbGrads[j] * learnRate;
                    h1Biases[j] += delta;
                    h1Biases[j] += h1PrevBiasesDelta[j] * momentum;
                    h1PrevBiasesDelta[j] = delta;
                }

                // 2.1 update hidden-to-hidden weights
                for (int j = 0; j < numHidden; ++j)
                {
                    for (int k = 0; k < numHidden; ++k)
                    {
                        double delta = hhGrads[j][k] * learnRate;
                        hhWeights[j][k] += delta;
                        hhWeights[j][k] += hhPrevWeightsDelta[j][k] * momentum;
                        hhPrevWeightsDelta[j][k] = delta;
                    }
                }

                // 2.2 update hidden 2 node biases
                for (int k = 0; k < numHidden; ++k)
                {
                    double delta = hhbGrads[k] * learnRate;
                    h2Biases[k] += delta;
                    h2Biases[k] += h2PrevBiasesDelta[k] * momentum;
                    h2PrevBiasesDelta[k] = delta;
                }

                // 3. update hidden-to-output weights
                for (int j = 0; j < numHidden; ++j)
                {
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = hoGrads[j][k] * learnRate;
                        hoWeights[j][k] += delta;
                        hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                        hoPrevWeightsDelta[j][k] = delta;
                    }
                }

                // 4. update output node biases
                for (int k = 0; k < numOutput; ++k)
                {
                    double delta = obGrads[k] * learnRate;
                    oBiases[k] += delta;
                    oBiases[k] += oPrevBiasesDelta[k] * momentum;
                    oPrevBiasesDelta[k] = delta;
                }

            } // each training item

        } // while
        //return this->GetWeights();
    } // Train

    private: void Shuffle(vector<int> sequence) // an instance method
    {
        //std::default_random_engine{}; //relatively casual, inexpert, and/or lightweight use.
        std::srand(time(NULL));
        std::random_shuffle(sequence.begin(), sequence.end());
    } // Shuffle

    private: double Error(vector<vector<double>> data)
    {
        // MSE == average squared error per training item
        double sumSquaredError = 0.0;
        vector<double> xValues(numInput);  // first numInput values in trainData
        vector<double> tValues(numOutput); // last numOutput values

        // walk thru each training case
        for (size_t i = 0; i < data.size(); ++i)
        {
            std::copy(data[i].begin(), data[i].begin() + numInput, xValues.begin());
            std::copy(data[i].begin() + numInput, data[i].begin() + numInput + numOutput, tValues.begin());

            vector<double> yValues = this->ComputeOutputs(xValues); // outputs using current weights

            for (int j = 0; j < numOutput; ++j)
            {
                double err = tValues[j] - yValues[j];
                sumSquaredError += err * err;
            }
        }
        return sumSquaredError / data.size();
    } // Error

    public: void ShowWeights()
    {
        vector<double> weights = this->GetWeights();
        std::cout << "weights: [" << endl;
        for(auto const &n : weights)
        {
            cout << ' ' << n;
        }cout << "]" << endl;
    }

}; // class NeuralNetwork
}

#endif