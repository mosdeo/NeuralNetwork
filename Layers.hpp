#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <typeinfo>
#include "Activation.hpp"
using namespace std;



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