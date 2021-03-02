#include "cudaError.h"

#include <hip/hip_runtime.h>

void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    hipError_t err = hipGetLastError();

    if (hipSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, hipGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}
