#include <cmath>
#include <vector>
using namespace std;

class Activation
{
    public: virtual vector<double> Forward(const vector<double>&)=0;
    public: virtual double Derivative(const double)=0;
};

class Tanh: public Activation
{
    public: vector<double> Forward(const vector<double>& nodeSum)
    {
        vector<double> result(nodeSum.size());
        

        for (size_t i = 0; i < nodeSum.size(); ++i)
        {
            result[i] = tanh(nodeSum[i]);
            cout << "result[i]=" << result[i] << endl;
        }

        return result;
    }

    public: double Derivative(const double x)
    {
        return 1 - pow(tanh(x), 2);
    }

    // public: vector<double> Derivative(const vector<double>& nodeSum)
    // {
    //     vector<double> result(nodeSum.size());

    //     for (size_t i = 0; i < nodeSum.size(); ++i)
    //     {
    //         result[i] = 1 - pow(tanh(nodeSum[i]), 2);
    //     }

    //     return result;
    // }
};