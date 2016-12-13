#include <cstdlib>
#include <iostream>
#include <algorithm> // std::random_shuffle
#include <chrono>
#include <ctime>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DrawData.hpp"
using namespace std;

namespace LKY
{
class NeuralNetwork
{
    private: int numInput; // number input nodes
    private: int numHidden;
    private: int numOutput;
    private: vector<double> inputs;
    private: vector<double> hiddens;
    private: vector<double> outputs;
    private: vector<vector<double>> ihWeights; // input-hidden
    private: vector<double> hBiases;
    private: vector<vector<double>> hoWeights; // hidden-output
    private: vector<double> oBiases;

    public: class Random
    {
        private: std::mt19937_64 randGen;
        public: Random()
        {
            randGen = std::mt19937_64();
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

        this->inputs.resize(numInput);
        this->hiddens.resize(numHidden);
        this->outputs.resize(numOutput);

        this->ihWeights = MakeMatrix(numInput, numHidden, 0.0);
        this->hBiases.resize(numHidden);

        this->hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
        this->oBiases.resize(numOutput);

        this->rnd = Random(seed);
        this->InitializeWeights(); // all weights and biases
    }                              // ctor

    private: static vector<vector<double>> MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
    {
        vector<double> row;
        row.assign(cols, v); //配置一個row的大小
        vector<vector<double>> array_2D;
        array_2D.assign(rows, row); //配置2維

        return array_2D;
    }

    private: void InitializeWeights() // helper for ctor
    {
        // initialize weights and biases to random values between 0.0001 and 0.001
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        vector<double> initialWeights(numWeights);
        double lo = -0.001;
        double hi = +0.001;

        for (int i = 0; i < numWeights; ++i)
        {
            //std::srand(std::time(0));
            initialWeights[i] = (hi-lo) *  rnd.NextDouble() + lo; // [-0.001 to +0.001]
                                                                           //initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
        }

        this->SetWeights(initialWeights);
    }

    public: void SetWeights(vector<double> weights)
    {
        // copy serialized weights and biases in weights[] array
        // to i-h weights, i-h biases, h-o weights, h-o biases
        size_t numWeights = (numInput * numHidden) +
                            (numHidden * numOutput) + numHidden + numOutput;

        if (weights.size() != numWeights)
        {
            cout << "throw new exception(\"Bad weights array in SetWeights\");" << endl;
            exit(EXIT_FAILURE);
        }

        int w = 0; // points into weights param

        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[w++];

        for (int j = 0; j < numHidden; ++j)
            hBiases[j] = weights[w++];

        for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
                hoWeights[j][k] = weights[w++];

        for (int k = 0; k < numOutput; ++k)
            //oBiases[k] = weights[k++];
            oBiases[k] = weights[w++];
    }

    public: vector<double> GetWeights()
    {
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        vector<double> result(numWeights);

        int w = 0;
        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                result[w++] = ihWeights[i][j];

        for (int j = 0; j < numHidden; ++j)
            result[w++] = hBiases[j];

        for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
                result[w++] = hoWeights[j][k];

        for (int k = 0; k < numOutput; ++k)
            result[w++] = oBiases[k];

        return result;
    }

    public: vector<double> ComputeOutputs(vector<double> xValues)
    {
        vector<double> hSums(numHidden); // hidden nodes sums scratch array
        vector<double> oSums(numOutput); // output nodes sums

        for (int i = 0; i < numInput; ++i) // copy x-values to inputs
            this->inputs[i] = xValues[i];

        for (int j = 0; j < numHidden; ++j) // compute i-h sum of weights * inputs
            for (int i = 0; i < numInput; ++i)
                hSums[j] += this->inputs[i] * this->ihWeights[i][j]; // note +=

        for (int j = 0; j < numHidden; ++j) // add biases to input-to-hidden sums
            hSums[j] += this->hBiases[j];

        for (int j = 0; j < numHidden; ++j)        // apply activation
            this->hiddens[j] = HyperTan(hSums[j]); // hard-coded

        for (int k = 0; k < numOutput; ++k) // compute h-o sum of weights * hOutputs
            for (int j = 0; j < numHidden; ++j)
                oSums[k] += hiddens[j] * hoWeights[j][k];

        for (int k = 0; k < numOutput; ++k) // add biases to input-to-hidden sums
            oSums[k] += oBiases[k];

        std::copy(oSums.begin(), oSums.begin() + this->numOutput, this->outputs.begin()); // copy without activation

        vector<double> retResult(numOutput); // could define a GetOutputs
        std::copy(this->outputs.begin(), this->outputs.begin() + this->numOutput, retResult.begin());

        return retResult;
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

    public: vector<double> Train(vector<vector<double>> trainData, int maxEpochs, double learnRate, double momentum)
    {
        // train using back-prop
        // back-prop specific arrays
        vector<vector<double>> hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weights gradients
        vector<double> obGrads(numOutput);                                      // output biases gradients

        vector<vector<double>> ihGrads = MakeMatrix(numInput, numHidden, 0.0); // input-to-hidden weights gradients
        vector<double> hbGrads(numHidden);                                     // hidden biases gradients

        vector<double> oSignals(numOutput); // signals == gradients w/o associated input terms
        vector<double> hSignals(numHidden); // hidden node signals

        // back-prop momentum specific arrays
        vector<vector<double>> ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
        vector<double> hPrevBiasesDelta(numHidden);
        vector<vector<double>> hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
        vector<double> oPrevBiasesDelta(numOutput);

        // train a back-prop style NN regression using learning rate and momentum
        int epoch = 0;
        vector<double> xValues(numInput);  // inputs
        vector<double> tValues(numOutput); // target values

        vector<int> sequence(trainData.size());
        for (size_t i = 0; i < trainData.size(); ++i)
            sequence[i] = i;

        while (epoch < maxEpochs)
        {
            ++epoch; // immediately to prevent display when 0

            int printInterval = maxEpochs/10; // interval to check validation data
            double trainErr = Error(trainData); //計算當下訓練誤差

            if (0 == epoch % printInterval)
            { //每 printInterval 次才顯示一次資訊
                
                cout << "epoch = " << epoch << "  training error = " << trainErr << endl;
            }

            if(true) //繪製訓練過程testData
            { 
                size_t numItems = 120;
                vector<vector<double>> testData(numItems, vector<double>(2));
                
                for(size_t i=0;i<numItems;i++)
                {//產生所有取樣點
                    testData[i][0] = i*(2*M_PI)/(double)numItems;
                    testData[i][1] = ComputeOutputs(testData[i])[0];
                }

                string strPngName = "png/訓練途中" + to_string(epoch) + ".png";
                string strPutText = "Epoch:"+to_string(epoch)+"/"+to_string(maxEpochs)+"  Err:" + to_string(trainErr);

                cv::imwrite(strPngName.c_str(),DrawData("訓練途中", testData, strPutText));
            }



            Shuffle(sequence); // visit each training data in random order
            for (size_t ii = 0; ii < trainData.size(); ++ii)
            {
                int idx = sequence[ii];

                std::copy(trainData[idx].begin(), trainData[idx].begin() + numInput, xValues.begin());
                std::copy(trainData[idx].begin() + numInput, trainData[idx].begin() + numInput + numOutput, tValues.begin());

                ComputeOutputs(xValues); // copy xValues in, compute outputs

                // indices: i = inputs, j = hiddens, k = outputs

                // 1. compute output nodes signals (assumes constant activation)
                for (int k = 0; k < numOutput; ++k)
                {
                    double derivative = 1.0; // for dummy output activation f'
                    oSignals[k] = (tValues[k] - outputs[k]) * derivative;//計算輸出與目標的差值
                }

                // 2. compute hidden-to-output weights gradients using output signals
                for (int j = 0; j < numHidden; ++j)
                    for (int k = 0; k < numOutput; ++k)
                        hoGrads[j][k] = oSignals[k] * hiddens[j];

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
                    double derivative = (1 + hiddens[j]) * (1 - hiddens[j]); // for tanh
                    hSignals[j] = sum * derivative;
                }

                // 4. compute input-hidden weights gradients
                for (int i = 0; i < numInput; ++i)
                    for (int j = 0; j < numHidden; ++j)
                        ihGrads[i][j] = hSignals[j] * inputs[i];

                // 4b. compute hidden node biases gradienys
                for (int j = 0; j < numHidden; ++j)
                    hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

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
                    hBiases[j] += delta;
                    hBiases[j] += hPrevBiasesDelta[j] * momentum;
                    hPrevBiasesDelta[j] = delta;
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
        return this->GetWeights();
    } // Train

    private: void Shuffle(vector<int> sequence) // an instance method
    {
        //std::default_random_engine{}; //relatively casual, inexpert, and/or lightweight use.
        std::mt19937{}; //it's ideal for scientific applications
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
