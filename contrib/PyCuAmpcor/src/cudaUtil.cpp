#include "cudaUtil.h"

#include <hip/hip_runtime_api.h>

int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(hipGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0 || devID > device_count-1)
    {
        fprintf(stderr, "gpuDeviceInit() Device %d is not a valid GPU device. \n", devID);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(hipSetDevice(devID));
    printf("gpuDeviceInit() Using CUDA Device %d ...\n", devID);

    return devID;
}

void gpuDeviceList()
{
    int device_count = 0;
    int current_device = 0;
    hipDeviceProp_t deviceProp;
    checkCudaErrors(hipGetDeviceCount(&device_count));

    fprintf(stderr, "Detecting all CUDA devices ...\n");
    if (device_count == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    while (current_device < device_count)
    {
        checkCudaErrors(hipGetDeviceProperties(&deviceProp, current_device));
// We only have to worry about this in GPU code
// (Not super sure but HIP-CPU doesn't define hipComputeModeProhibited)
// This seems to be CUDA-specific so let's just skip it.
#ifdef __CUDACC__
        if (deviceProp.computeMode == hipComputeModeProhibited)
        {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: device is running in <Compute Mode Prohibited> \n", current_device, deviceProp.name);
        }
        else if (deviceProp.major < 1)
        {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is not available: device does not support CUDA \n", current_device, deviceProp.name);
        }
        else {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is available.\n", current_device, deviceProp.name);
        }
#endif
        current_device++;
    }
}

