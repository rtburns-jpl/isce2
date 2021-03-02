/******************************************************************************
* Copyright (c) 2016 - 2021 Advanced Micro Devices, Inc. All rights reserved.
* Copyright (c) 2021 Jet Propulsion Laboratory, Caltech. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#pragma once

#include <fftw3.h>

#include <hip/hip_complex.h>

#define HIPFFT_FORWARD FFTW_FORWARD
#define HIPFFT_BACKWARD FFTW_BACKWARD

using hipfftComplex = hipComplex;
using hipfftDoubleComplex = hipDoubleComplex;
using hipfftReal = float;
using hipfftDoubleReal = double;

// TODO can we use threads for this?
#define hipfftSetStream(...)

/*
inline hipfftComplex& operator+=(hipfftComplex& lhs, const float2& rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

inline hipfftComplex& operator*=(hipfftComplex& lhs, const float2& rhs) {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}
*/

enum hipfftResult {
    HIPFFT_SUCCESS = 0,
};

enum hipfftType {
    HIPFFT_R2C,
    HIPFFT_C2R,
    HIPFFT_C2C,
};

struct hipfftHandle {
    fftw_plan plan;
    int       rank;
    int*      n;
    int*      inembed;
    int       istride;
    int       idist;
    int*      onembed;
    int       ostride;
    int       odist;
    int       batch;
};

inline
hipfftResult hipfftDestroy(hipfftHandle& plan) {
    return HIPFFT_SUCCESS;
}

inline
hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch) {

    plan->rank = rank;
    plan->n = n;
    plan->inembed = inembed;
    plan->istride = istride;
    plan->idist = idist;
    plan->onembed = onembed;
    plan->ostride = ostride;
    plan->odist = odist;
    plan->batch = batch;
    return HIPFFT_SUCCESS;
}

inline
hipfftResult hipfftExecC2C(hipfftHandle   plan,
                           hipfftComplex* idata,
                           hipfftComplex* odata,
                           int            direction) {

    plan.plan = fftw_plan_many_dft(
            plan.rank, plan.n, plan.batch,
            (fftw_complex*) idata, plan.inembed, plan.istride, plan.idist,
            (fftw_complex*) odata, plan.onembed, plan.ostride, plan.odist,
            direction, FFTW_ESTIMATE);

    fftw_execute(plan.plan);
    fftw_destroy_plan(plan.plan);
    return HIPFFT_SUCCESS;
}

inline
hipfftResult hipfftExecR2C(
        hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata) {

    plan.plan = fftw_plan_many_dft_r2c(
            plan.rank, plan.n, plan.batch,
            (double*) idata, plan.inembed, plan.istride, plan.idist,
            (fftw_complex*) odata, plan.onembed, plan.ostride, plan.odist,
            FFTW_ESTIMATE);

    fftw_execute(plan.plan);
    fftw_destroy_plan(plan.plan);
    return HIPFFT_SUCCESS;
}

inline
hipfftResult hipfftExecC2R(
        hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata) {

    plan.plan = fftw_plan_many_dft_c2r(
            plan.rank, plan.n, plan.batch,
            (fftw_complex*) idata, plan.inembed, plan.istride, plan.idist,
            (double*) odata, plan.onembed, plan.ostride, plan.odist,
            FFTW_ESTIMATE);

    fftw_execute(plan.plan);
    fftw_destroy_plan(plan.plan);
    return HIPFFT_SUCCESS;
}
