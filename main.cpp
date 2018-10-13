/*
 * Tyler Filla
 * CS 4340
 * Project 3
 */

#include <cmath>
#include <iostream>

static const double DATASET[8][2] = {
    {1, -1}, // y = 0
    {2, 1},
    {3, -1}, // y = 0
    {4, 1},
    {5, -1}, // y = 0
    {6, 1},
    {7, 1},
    {8, 1},
};

int main(int argc, char* argv[])
{
    // Sample count
    const int N = 8;

    std::cout << "Sample size: " << N << "\n";

    // Fixed learning rate
    const double lr = 0.1;

    std::cout << "Fixed learning rate: " << lr << "\n";

    // Fixed iteration limit
    const int LIMIT = 1000000;

    std::cout << "Fixed iteration limit: " << LIMIT << "\n";

    // Initialize weight vector "w"
    // The first dimension of w provides constant bias
    double w0 = 0;
    double w1 = 0;

    std::cout << "Initial weights: (" << w0 << ", " << w1 << ")\n";

    int t;
    for (t = 0; t < LIMIT; ++t)
    {
        // Gradient vector
        double g0 = 0;
        double g1 = 0;

        for (int n = 0; n < N; ++n)
        {
            // Feature vector "x"
            // The first dimension of x is always tied to one
            // This serves as a pseudo-dimension that passes through w0
            const double x0 = 1;
            double x1 = DATASET[n][0];

            // Result vector "y"
            double y = DATASET[n][1];

            // Temporary numerator vector
            double a0 = y * x0;
            double a1 = y * x1;

            // Denominator value
            double b = 1 + std::exp(y * (w0 * x0 + w1 * x1));

            // Add intermediate quotient to almost-gradient vector
            g0 += a0 / b;
            g1 += a1 / b;
        }

        // Finish gradient vector computation
        g0 *= (-1.0 / N);
        g1 *= (-1.0 / N);

        // Magnitude of gradient vector
        double g_mag = g0 * g0 + g1 * g1;

        // If gradient vector is sufficiently small
        if (g_mag <= 0.001 * 0.001)
        {
            // Then stop
            std::cout << "Minimized gradient: " << g_mag << " magnitude\n";
            break;
        }

        // Find direction of descent
        double v0 = -g0;
        double v1 = -g1;

        // Update weights
        w0 += lr * v0;
        w1 += lr * v1;
    }

    std::cout << "\n";
    std::cout << t << " iteration(s) performed\n";
    std::cout << "Final weights: (" << w0 << ", " << w1 << ")\n";

    return 0;
}
