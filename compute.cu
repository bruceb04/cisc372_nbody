// Bruce Bermel & Paul Healy

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

// Kernel 1: compute acceleration from all bodies in subset 
// acceleration[i*NUMENTITIES + j] = from j to i 

__global__ void acceleration_computation(vector3 *acceleration, const vector3 *position, const double *mass) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // target body
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // source body

    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // no acceleration if its = itself
    if (i == j) {

        FILL_VECTOR(acceleration[i * NUMENTITIES + j], 0.0, 0.0, 0.0);
        return;

    }

    double dx = position[i][0] - position[j][0];
    double dy = position[i][1] - position[j][1];
    double dz = position[i][2] - position[j][2];

    double mag_square = dx*dx + dy*dy + dz*dz;

    //avoid division by zero id two bodies overlap
    if (mag_square == 0.0) {
	    FILL_VECTOR(acceleration[i * NUMENTITIES + j], 0.0, 0.0, 0.0);
	    return;
    }

    double magnitude = sqrt(mag_square);

    // a = G * m_j / r^2 (negative for attraction)
    double magA = GRAV_CONSTANT * (mass[j] / mag_square) * -1.0;

    FILL_VECTOR(
	acceleration[i * NUMENTITIES + j],
	dx * magA / magnitude,
	dy * magA / magnitude,
	dz * magA / magnitude
    );

}

// Kernel 2: sum over j for each i, then update velocity and position of body i 

__global__ void row_summation(const vector3 *acceleration, vector3 *velocity, vector3 *position) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i >= NUMENTITIES) { 
	return;
   }

    double ax = 0.0, ay = 0.0, az = 0.0;

    // Sum contributions from all j to body i
    for (int j = 0; j < NUMENTITIES; j++) {
	    const vector3 &a_ij = acceleration[i * NUMENTITIES + j];
	    ax += a_ij[0];
	    ay += a_ij[1];
	    az += a_ij[2];
    }

    // Euler integration
    velocity[i][0] += INTERVAL * ax;
    velocity[i][1] += INTERVAL * ay;
    velocity[i][2] += INTERVAL * az;

    position[i][0] += INTERVAL * velocity[i][0];
    position[i][1] += INTERVAL * velocity[i][1];
    position[i][2] += INTERVAL * velocity[i][2];

}

// one timestep
void compute(vector3 *d_acceleration, vector3 *d_velocity, vector3 *d_position, double *d_mass) {
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
