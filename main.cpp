#include "LKYDeepNN.hpp"


int main()
{
    int numEachHiddenNodes = 2;
    int numHiddenLayers = 2;
    LKYDeepNN nn(2, vector<int>(numHiddenLayers, numEachHiddenNodes), 2);

    vector<double> outputArray = nn.ForwardPropagation(vector<double>{2,2});

    
    for (double const output : outputArray)
    {//print
        printf("%lf, ",output);
    }cout << endl;
    cout << "順傳遞測試完成" <<endl;

    for (double const output : outputArray)
    {//print
        printf("%lf, ",output);
    }cout << endl;
    cout << "順傳遞測試完成" <<endl;

    nn.Training(10, vector<double>(2,0));

    for (double const output : outputArray)
    {//print
        printf("%lf, ",output);
    }cout << endl;
    cout << "順傳遞測試完成" <<endl;
}