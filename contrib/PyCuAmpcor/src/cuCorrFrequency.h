/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

// code guard
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H

#include <hip/hip_runtime.h>

// dependencies
#include "cudaUtil.h"
#include "cuArrays.h"

#include <hipfft.h>

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    hipfftHandle forwardPlan;
    hipfftHandle backwardPlan;
    // work data
    cuArrays<float2> *workFM;
    cuArrays<float2> *workFS;
    cuArrays<float> *workT;
    // cuda stream
    hipStream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, hipStream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results);
};

#endif //__CUCORRFREQUENCY_H
// end of file
