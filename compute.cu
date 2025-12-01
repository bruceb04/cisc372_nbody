// Bruce Bermel & Paul Healy

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

// Kernel 1: compute acceleration from all bodies in subset 
// acceleration[i*NUMENTITIES + j] = from j to i 

__global__ void acceleration_computation(v3 *acceleration, const v3 *position, const double *mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // target body
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // source body

    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // no acceleration if its = itself
    if (i == j) {

        FILL_VECTOR(acceleration[i * NUMENTITIES + j], 0.0, 0.0, 0.0);
        return;
    }

    v3 dist;
    double mag_square = 0.0;

    // dist = position[i] - position[j]
    for (int k = 0; k < 3; ++k) { 
        dist[k] = position[i][k] - position[j][k];
        mag_square += dist[k] * dist[k];
    }

    //avoid division by zero if two bodies overlap 
    if (mag_square == 0.0) {
        FILL_VECTOR(acceleration[i * NUMENTITIES + j], 0.0, 0.0, 0.0);
        return;
    }
    double magnitude = sqrt(mag_square);

    // a = G * m_j / r^2
    double magA = GRAV_CONSTANT * (mass[j] / mag_square) * -1.0;

    FILL_VECTOR(
        acceleration[i * NUMENTITIES + j],
        dist[0] * magA / magnitude,
        dist[1] * magA / magnitude,
        dist[2] * magA / magnitude
    );
}

// Kernel 2: sum over j for each i, then update velocity and position of body i 

__global__ void row_summation(const v3 *acceleration, v3 *velocity, v3 *position) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) return;

    v3 total_acceleration = {0.0, 0.0, 0.0};

    // Sum contributions
    for (int j = 0; j < NUMENTITIES; ++j) {
        for (int k = 0; k < 3; ++k) {
            total_acceleration[k] += acceleration[i * NUMENTITIES + j][k];
        }
    }

    // Simple explicit Euler integration step (ChatGPT helped with this part)
    for (int k = 0; k < 3; ++k) {
        velocity[i][k] += INTERVAL * total_acceleration[k];
        position[i][k] += INTERVAL * velocity[i][k];
    }
}

// one timestep
void compute(v3 *d_acceleration, v3 *d_velocity, v3 *d_position, double *d_mass) {
    // --- Launch config for pairwise acceleration kernel ---
    dim3 blockDimAcc(16, 16);
    dim3 gridDimAcc(
        (NUMENTITIES + blockDimAcc.x - 1) / blockDimAcc.x,
        (NUMENTITIES + blockDimAcc.y - 1) / blockDimAcc.y
    );

    acceleration_computation<<<gridDimAcc, blockDimAcc>>>(
        d_acceleration, d_position, d_mass
    );

    // --- Launch config for row summation kernel (1D) ---
    int blockSize = 256;
    int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

    row_summation<<<numBlocks, blockSize>>>(
        d_acceleration, d_velocity, d_position
    );
}
