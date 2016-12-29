#include <tuple>
using namespace std;

double randUniform(double a, double b)
{
  //return Math.random() * (b - a) + a;
  return (rand()/(double)RAND_MAX)* (b - a) + a;
}

vector<vector<double>> classifyCircleData(int numSamples=80, double noise=0)
{
    vector<vector<double>> points(0, vector<double>(3));
    double radius = 5;

    auto getCircleLabel = [](std::tuple<double, double> p, std::tuple<double, double> center, double radius)
    {
        double dist_p_to_center = pow((get<0>(p) - get<0>(center)),2) + pow((get<1>(p) - get<1>(center)),2);
        dist_p_to_center = pow(dist_p_to_center, 0.5);
        return (dist_p_to_center < (radius * 0.5)) ? 1 : -1;
    };

    // Generate positive points inside the circle.
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = randUniform(0, radius * 0.5);
        double angle = randUniform(0, 2 * M_PI);
        double x = r * sin(angle);
        double y = r * cos(angle);

        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        points.push_back({x, y, (double)label});
    }

    // // Generate negative points outside the circle.
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = randUniform(radius * 0.7, radius);
        double angle = randUniform(0, 2 * M_PI);
        double x = r * sin(angle);
        double y = r * cos(angle);
        
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        points.push_back({x, y, (double)label});
    }

    return points;
}