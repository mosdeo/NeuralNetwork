#include "LKYDeepNN.hpp"


int main()
{
    int numEachHiddenNodes = 2;
    int numHiddenLayers = 1;
    LKYDeepNN nn(2, vector<int>(numHiddenLayers, numEachHiddenNodes), 2);

    vector<double> outputArray = nn.ForwardPropagation(vector<double>{2,2});

    //print
    for (double const output : outputArray)
    {
        printf("%lf, ",output);
    }cout << endl;
}