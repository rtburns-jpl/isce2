/*
 * @file cuAmpcorUtil.h
 * @brief Header file to include various routines for cuAmpcor
 *
 *
 */

// code guard
#ifndef __CUAMPCORUTIL_H
#define __CUAMPCORUTIL_H

#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cudaError.h"
#include "debug.h"
#include "cudaUtil.h"
#include "float2.h"


//in cuArraysCopy.cu: various utilities for copy images file in gpu memory
void cuArraysCopyToBatch(cuArrays<float2> *image1, cuArrays<float2> *image2, int strideH, int strideW, hipStream_t stream);
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW, hipStream_t stream);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW, hipStream_t stream);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW, hipStream_t stream);
void cuArraysCopyC2R(cuArrays<float2> *image1, cuArrays<float> *image2, int strideH, int strideW, hipStream_t stream);

// same routine name overloaded for different data type
// extract data from a large image
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int2> *offset, hipStream_t stream);
void cuArraysCopyExtract(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, int2 offset, hipStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float> *imagesOut, int2 offset, hipStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int2 offset, hipStream_t stream);
void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, cuArrays<int2> *offsets, hipStream_t stream);
void cuArraysCopyExtract(cuArrays<float3> *imagesIn, cuArrays<float3> *imagesOut, int2 offset, hipStream_t stream);

void cuArraysCopyInsert(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut, int offsetX, int offersetY, hipStream_t stream);
void cuArraysCopyInsert(cuArrays<float3> *imageIn, cuArrays<float3> *imageOut, int offsetX, int offersetY, hipStream_t stream);
void cuArraysCopyInsert(cuArrays<float> *imageIn, cuArrays<float> *imageOut, int offsetX, int offsetY, hipStream_t stream);
void cuArraysCopyInsert(cuArrays<int> *imageIn, cuArrays<int> *imageOut, int offsetX, int offersetY, hipStream_t stream);

void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut,hipStream_t stream);
void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float2> *imageOut,hipStream_t stream);
void cuArraysCopyPadded(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut,hipStream_t stream);
void cuArraysSetConstant(cuArrays<float> *imageIn, float value, hipStream_t stream);

void cuArraysR2C(cuArrays<float> *image1, cuArrays<float2> *image2, hipStream_t stream);
void cuArraysC2R(cuArrays<float2> *image1, cuArrays<float> *image2, hipStream_t stream);
void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2, hipStream_t stream);

// cuDeramp.cu: deramping phase
void cuDeramp(int method, cuArrays<float2> *images, hipStream_t stream);
void cuDerampMethod1(cuArrays<float2> *images, hipStream_t stream);

// cuArraysPadding.cu: various utilities for oversampling padding
void cuArraysPadding(cuArrays<float2> *image1, cuArrays<float2> *image2, hipStream_t stream);
void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2, hipStream_t stream);

//in cuCorrNormalization.cu: utities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<float> *images, hipStream_t stream);
void cuCorrNormalize(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, hipStream_t stream);

//in cuOffset.cu: utitilies for determining the max locaiton of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cuArrays<float> *maxval, hipStream_t stream);
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, hipStream_t stream);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn, cuArrays<float2> *offsetFinal,
                      int OverSampleRatioZoomin, int OverSampleRatioRaw,
                      int xHalfRangeInit,  int yHalfRangeInit,
                      hipStream_t stream);

void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
        int xOldRange, int yOldRange, int xNewRange, int yNewRange, hipStream_t stream);

//in cuCorrTimeDomain.cu: cross correlation in time domain
void cuCorrTimeDomain(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results, hipStream_t stream);

//in cuCorrFrequency.cu: cross correlation in freq domain, also include fft correlatior class
void cuArraysElementMultiply(cuArrays<float2> *image1, cuArrays<float2> *image2, hipStream_t stream);
void cuArraysElementMultiplyConjugate(cuArrays<float2> *image1, cuArrays<float2> *image2, float coef, hipStream_t stream);


// For SNR estimation on Correlation surface (Minyan Zhong)
// implemented in cuArraysCopy.cu
void cuArraysCopyExtractCorr(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, hipStream_t stream);
// implemented in cuCorrNormalization.cu
void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum, cuArrays<int> *imagesValidCount, hipStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue, hipStream_t stream);

// implemented in cuEstimateStats.cu
void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, int templateSize, cuArrays<float3> *covValue, hipStream_t stream);

#endif

// end of file
