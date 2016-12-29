#include <tuple>
#include <random>
using namespace std;

double randUniform(double a, double b)
{
  return (rand()/(double)RAND_MAX)* (b - a) + a;
}

vector<vector<double>> classifyCircleData(int numSamples=80, double noise=0.1)
{
    vector<vector<double>> points(0, vector<double>(3));
    double radius = 5;

    auto getCircleLabel = [](std::tuple<double, double> p, std::tuple<double, double> center, double radius)
    {
        double dist_p_to_center = pow((get<0>(p) - get<0>(center)),2) + pow((get<1>(p) - get<1>(center)),2);
        dist_p_to_center = pow(dist_p_to_center, 0.5);
        return (dist_p_to_center < (radius * 0.5)) ? 1 : -1;
    };

    //std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(0);    // random-number engine used (Mersenne-Twister in this case)
    //std::minstd_rand0 rng();
    std::uniform_real_distribution<double> uni_r; // guaranteed unbiased
    std::uniform_real_distribution<double> uni_angle(0, 2 * M_PI); // guaranteed unbiased
    std::uniform_real_distribution<double> uni_noise(-radius, radius); // guaranteed unbiased
    //auto random_integer = uni(rng);

    // Generate positive points inside the circle.
    uni_r = std::uniform_real_distribution<double>(0, radius * 0.5);
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = uni_r(rng);
        double angle = uni_angle(rng);
        double x = r * sin(angle);
        double y = r * cos(angle);

        double noiseX = uni_noise(rng) * noise;
        double noiseY = uni_noise(rng) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        points.push_back({x, y, (double)label});
    }

    // // Generate negative points outside the circle.
    uni_r = std::uniform_real_distribution<double>(radius * 0.7, radius);
    for (int i = 0; i < numSamples / 2; i++)
    {
        double r = uni_r(rng);
        double angle = uni_angle(rng);
        double x = r * sin(angle);
        double y = r * cos(angle);
        
        double noiseX = uni_noise(rng) * noise;
        double noiseY = uni_noise(rng) * noise;
        std::tuple<double, double> noise(x+noiseX, y+noiseY);

        int label = getCircleLabel(noise, std::tuple<double, double>(0,0),radius);
        points.push_back({x, y, (double)label});
    }

    return points;
}