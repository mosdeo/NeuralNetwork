#include <cmath>

class LossFunction
{
    public: virtual double GetCost(double tragetValue, double currentValue)=0;
};

class CrossEntropy: public LossFunction
{
    public: CrossEntropy()
    {
        cout << "LossFunction is CrossEntropy." << endl;
    }

    public: double GetCost(double tragetValue, double currentValue)
    {
        if(-1 == tragetValue)
        {
            tragetValue=0;
        }
        return (tragetValue*log(currentValue) + (1-tragetValue)*log(1-currentValue));
    }
};

class Diff: public LossFunction
{
    public: Diff()
    {
        cout << "LossFunction is Diff." << endl;
    }

    public: double GetCost(double tragetValue, double currentValue)
    {
        return tragetValue-currentValue;
    }
};

class DiffSqrt: public LossFunction
{
    public: DiffSqrt()
    {
        cout << "LossFunction is DiffSqrt." << endl;
    }

    public: double GetCost(double tragetValue, double currentValue)
    {
        return -pow(tragetValue - currentValue, 2);
    }
};