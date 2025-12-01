// Bruce Bermel & Paul Healy

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// Global host pointers
vector3 *hVel;
vector3 *hPos;
double  *mass;

// Global device pointers
vector3 *d_hVel;
vector3 *d_hPos;
double  *d_mass;
vector3 *d_acc;                // size = NUMENTITIES*NUMENTITIES

// chatGPT cuda check
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// create memory for number of objects in system
void initHostMemory()
{
    hVel = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES);
    hPos = (vector3 *)malloc(sizeof(vector3) * NUMENTITIES);
    mass = (double  *)malloc(sizeof(double)  * NUMENTITIES);

    if (!hVel || !hPos || !mass) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

// free initially hosted memory
void freeHostMemory()
{
    free(hVel);
    free(hPos);
    free(mass);
}

// initDeviceMemory: allocate device memory for positions, velocities, masses, and accelerations
void initDeviceMemory()
{
    size_t vecSize   = sizeof(vector3) * NUMENTITIES;
    size_t massSize  = sizeof(double)  * NUMENTITIES;
    size_t accSize   = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    CUDA_CHECK(cudaMalloc((void **)&d_hVel, vecSize));
    CUDA_CHECK(cudaMalloc((void **)&d_hPos, vecSize));
    CUDA_CHECK(cudaMalloc((void **)&d_mass, massSize));
    CUDA_CHECK(cudaMalloc((void **)&d_acc,  accSize));
}

// freeDeviceMemory: free all device allocations
void freeDeviceMemory()
{
    CUDA_CHECK(cudaFree(d_hVel));
    CUDA_CHECK(cudaFree(d_hPos));
    CUDA_CHECK(cudaFree(d_mass));
    CUDA_CHECK(cudaFree(d_acc));
}

// copyHostToDevice: upload initial state
void copyHostToDevice()
{
    size_t vecSize  = sizeof(vector3) * NUMENTITIES;
    size_t massSize = sizeof(double)  * NUMENTITIES;

    CUDA_CHECK(cudaMemcpy(d_hVel, hVel, vecSize,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hPos, hPos, vecSize,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, mass, massSize, cudaMemcpyHostToDevice));
}

// copyDeviceToHost: download final state (for printing / saving)
void copyDeviceToHost()
{
    size_t vecSize  = sizeof(vector3) * NUMENTITIES;
    size_t massSize = sizeof(double)  * NUMENTITIES;

    CUDA_CHECK(cudaMemcpy(hVel, d_hVel, vecSize,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPos, d_hPos, vecSize,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mass, d_mass, massSize, cudaMemcpyDeviceToHost));
}

// planetFill: Fill the first NUMPLANETS+1 entries with Sun + planets
void planetFill(){
    int i,j;
    double data[][7] = {
        SUN, MERCURY, VENUS, EARTH, MARS,
        JUPITER, SATURN, URANUS, NEPTUNE
    };

    for (i = 0; i <= NUMPLANETS; i++) {
        for (j = 0; j < 3; j++) {
            hPos[i][j] = data[i][j];
            hVel[i][j] = data[i][j + 3];
        }
        mass[i] = data[i][6];
    }
}

// randomFill: Fill the rest of the objects randomly starting at index start
void randomFill(int start, int count)
{
    int i, j;
    for (i = start; i < start + count; i++) {
        for (j = 0; j < 3; j++) {
            // NOTE: your original code had distance/velocity swapped in comments;
            // keeping same formulae, just cleaning up structure.
            hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2.0 - MAX_DISTANCE;
            hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2.0 - MAX_VELOCITY;
        }
        // mass only needs to be set once per body
        mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
    }
}

// printSystem: Prints out the entire system to the supplied file
void printSystem(FILE* handle){
    int i,j;
    for (i = 0; i < NUMENTITIES; i++) {
        fprintf(handle, "pos=(");
        for (j = 0; j < 3; j++) {
            fprintf(handle, "%lf,", hPos[i][j]);
        }
        fprintf(handle, "),v=(");                     // fixed: was printf before
        for (j = 0; j < 3; j++) {
            fprintf(handle, "%lf,", hVel[i][j]);
        }
        fprintf(handle, "),m=%lf\n", mass[i]);
    }
}

int main(int argc, char **argv)
{
    clock_t t0 = clock();
    int t_now;

    srand(1234);
    initHostMemory(NUMENTITIES);

    // Initialize host-side system
    planetFill();
    randomFill(NUMPLANETS + 1, NUMASTEROIDS);

#ifdef DEBUG
    // Print initial system (host side)
    printSystem(stdout);
#endif

    // Initialize and copy to device
    initDeviceMemory(NUMENTITIES);
    copyHostToDevice(NUMENTITIES);

    // Main simulation loop: all heavy work is in GPU compute()
    // compute() is assumed to be the CUDA step we wrote earlier:
    //     void compute(vector3 *d_acc, vector3 *d_vel, vector3 *d_pos, double *d_mass);
    for (t_now = 0; t_now < DURATION; t_now += INTERVAL) {
        compute(d_acc, d_hVel, d_hPos, d_mass);
        // If you want to check for errors per step in debug builds:
        // CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Bring final state back to host
    copyDeviceToHost(NUMENTITIES);

    clock_t t1 = clock() - t0;

#ifdef DEBUG
    // Print final system (host side, after GPU simulation)
    printSystem(stdout);
#endif

    printf("This took a total time of %f seconds\n",
           (double)t1 / CLOCKS_PER_SEC);

    freeDeviceMemory();
    freeHostMemory();

    return 0;
}
