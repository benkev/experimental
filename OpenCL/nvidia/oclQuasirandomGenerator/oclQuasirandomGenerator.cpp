/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

///////////////////////////////////////////////////////////////////////////////
// This sample implements Niederreiter quasirandom number generator
// and Moro's Inverse Cumulative Normal Distribution generator
///////////////////////////////////////////////////////////////////////////////

#include "oclQuasirandomGenerator_common.h"

// forward declarations
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]
    );
extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int x);

// OpenCL wrappers
void QuasirandomGeneratorGPU(cl_command_queue cqCommandQueue,
			                 cl_kernel ckQuasirandomGenerator,
                             cl_mem d_Output,
                             cl_mem c_Table,
                             unsigned int seed,
                             unsigned int N,
                             size_t szWgXDim);
void InverseCNDGPU(cl_command_queue cqCommandQueue, 
		   cl_kernel ckInverseCNDGPU, 
		   cl_mem d_Output, 
		   unsigned int pathN,
           unsigned int iDevice,
           unsigned int nDevice,
           size_t szWgXDim);

// size of output random array
unsigned int N = 1048576;

///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    cl_context cxGPUContext;                          // OpenCL context
    cl_command_queue cqCommandQueue[MAX_GPU_COUNT];   // OpenCL command que
    cl_platform_id cpPlatform;                        // OpenCL platform
    cl_uint nDevice;                                  // OpenCL device count
    cl_device_id *cdDevices;                          // OpenCL device list    
    cl_program cpProgram;                             // OpenCL program
    cl_kernel ckQuasirandomGenerator, ckInverseCNDGPU;// OpenCL kernel
    cl_mem *d_Output, *c_Table;                       // OpenCL buffers
    float *h_OutputGPU;
    cl_int ciErr;                                     // Error code var
    unsigned int dim, pos;
    double delta, ref, sumDelta, sumRef, L1norm;
    unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];
    bool bPassFlag = false;

    // Get the devices
    ciErr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &nDevice);
    cdDevices = (cl_device_id *)malloc(nDevice * sizeof(cl_device_id) );
    ciErr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, nDevice, cdDevices, \
                           NULL);

    //Create the context
    cxGPUContext = clCreateContext(0, nDevice, cdDevices, NULL, NULL, &ciErr);
    //r oclCheckErrorEX(ciErr, CL_SUCCESS, NULL);
    
    cl_uint id_device;

         cdDevices[0] = cdDevices[id_device];

        // create a command que
        cqCommandQueue[0] = clCreateCommandQueue(cxGPUContext, cdDevices[0], \
                                                 0, &ciErr);
        nDevice = 1;   




    d_Output = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    c_Table = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    for (cl_uint i = 0; i < nDevice; i++)
    {
        d_Output[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, QRNG_DIMENSIONS * N / nDevice * sizeof(cl_float), NULL, &ciErr);
        oclCheckErrorEX(ciErr, CL_SUCCESS, NULL);
    }
    h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(cl_float));

    shrLog("Initializing QRNG tables...\n");
    initQuasirandomGenerator(tableCPU);
    for (cl_uint i = 0; i < nDevice; i++)
    {
        c_Table[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int), 
    		     NULL, &ciErr);
        ciErr |= ciErr;
        ciErr |= clEnqueueWriteBuffer(cqCommandQueue[i], c_Table[i], CL_TRUE, 0, 
            QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int), tableCPU, 0, NULL, NULL);
    }
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL);

    shrLog("Create and build program...\n");
    size_t szKernelLength; // Byte size of kernel code
    char *progSource = oclLoadProgSource(shrFindFilePath("QuasirandomGenerator.cl", argv[0]), "// My comment\n", &szKernelLength);
	oclCheckErrorEX(progSource == NULL, false, NULL);

    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&progSource, &szKernelLength, &ciErr);
    ciErr |= clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, (double)ciErr, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "QuasirandomGenerator.ptx");
        oclCheckError(ciErr, CL_SUCCESS); 
    }

    shrLog("Create QuasirandomGenerator kernel...\n"); 
    ckQuasirandomGenerator = clCreateKernel(cpProgram, "QuasirandomGenerator", &ciErr);
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 

    shrLog("Create InverseCND kernel...\n\n"); 
    ckInverseCNDGPU = clCreateKernel(cpProgram, "InverseCND", &ciErr);
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 

    shrLog(">>>Launch QuasirandomGenerator kernel...\n\n"); 

    // determine work group sizes for each device
	size_t* szWorkgroup = new size_t[nDevice]; 
	for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
	{
        ciErr |= clGetKernelWorkGroupInfo(ckQuasirandomGenerator, cdDevices[iDevice], 
                                        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szWorkgroup[iDevice], NULL);
      
		szWorkgroup[iDevice] = 64 * ((szWorkgroup[iDevice] / QRNG_DIMENSIONS)/64);
    }
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL);

#ifdef GPU_PROFILING
    int numIterations = 100;
    for (int i = -1; i< numIterations; i++)
    {
		if (i == 0)
		{
			for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
			{
				clFinish(cqCommandQueue[iDevice]);
			}
			shrDeltaT(1);
		}
#endif
 
        for (cl_uint i = 0; i < nDevice; i++)
        {
            QuasirandomGeneratorGPU(cqCommandQueue[i], ckQuasirandomGenerator, d_Output[i], c_Table[i], 0, N/nDevice, szWorkgroup[i]);    
        }

#ifdef GPU_PROFILING
    }
    for (cl_uint i = 0; i < nDevice; i++)
    {
        clFinish(cqCommandQueue[i]);
    }
    double gpuTime = shrDeltaT(1)/(double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclQuasirandomGenerator, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n", 
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS * N, nDevice, szWorkgroup[0]);
#endif

    shrLog("\nRead back results...\n"); 
    int offset = 0;
    for (cl_uint i = 0; i < nDevice; i++)
    {
        ciErr |= clEnqueueReadBuffer(cqCommandQueue[i], d_Output[i], CL_TRUE, 0, sizeof(cl_float) * QRNG_DIMENSIONS * N / nDevice, 
            h_OutputGPU + offset, 0, NULL, NULL);
        offset += QRNG_DIMENSIONS * N / nDevice;
    }
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 

    shrLog("Comparing to the CPU results...\n\n");
    sumDelta = 0;
    sumRef   = 0;
    for (cl_uint i = 0; i < nDevice; i++)
    {
        for(dim = 0; dim < QRNG_DIMENSIONS; dim++)
        {
            for(pos = 0; pos < N / nDevice; pos++) 
            {
	            ref       = getQuasirandomValue63(pos, dim);
	            delta     = (double)h_OutputGPU[i*QRNG_DIMENSIONS*N/nDevice + dim * N / nDevice + pos] - ref;
	            sumDelta += fabs(delta);
	            sumRef   += fabs(ref);
	        }
        }
    }
    L1norm = sumDelta / sumRef;
    shrLog("  L1 norm: %E\n", L1norm);
    shrLog("  ckQuasirandomGenerator deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag = (L1norm < 1e-6);

    shrLog(">>>Launch InverseCND kernel...\n\n"); 

    // determine work group sizes for each device
	for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
	{
        ciErr |= clGetKernelWorkGroupInfo(ckInverseCNDGPU, cdDevices[iDevice], 
                                        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szWorkgroup[iDevice], NULL);
        if (szWorkgroup[iDevice] >= 128)
        {
            szWorkgroup[iDevice] = 128;
        }
        else
        {
            szWorkgroup[iDevice] = 64 * (szWorkgroup[iDevice] / 64);
        }
    }
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL);

#ifdef GPU_PROFILING
    for (int i = -1; i< numIterations; i++)
    {
		if (i == 0) 
		{
			for (cl_uint iDevice = 0; iDevice < nDevice; iDevice++)
			{
				clFinish(cqCommandQueue[iDevice]);
			}
			shrDeltaT(1);
		}
#endif
        for (cl_uint i = 0; i < nDevice; i++)
        {
            InverseCNDGPU(cqCommandQueue[i], ckInverseCNDGPU, d_Output[i], QRNG_DIMENSIONS * N / nDevice, i, nDevice, szWorkgroup[i]);    
        }
#ifdef GPU_PROFILING
    }

	for (cl_uint i = 0; i < nDevice; i++)
    {
        clFinish(cqCommandQueue[i]);
    }
    gpuTime = shrDeltaT(1)/(double)numIterations;
    shrLogEx(LOGBOTH | MASTER, 0, "oclQuasirandomGenerator-inverse, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n", 
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS * N, nDevice, szWorkgroup[0]);    
#endif

    shrLog("\nRead back results...\n"); 
    offset = 0;
    for (cl_uint i = 0; i < nDevice; i++)
    {
        ciErr |= clEnqueueReadBuffer(cqCommandQueue[i], d_Output[i], CL_TRUE, 0, 
            sizeof(cl_float) * QRNG_DIMENSIONS * N / nDevice, h_OutputGPU + offset, 0, NULL, NULL);
        offset += QRNG_DIMENSIONS * N / nDevice;
        oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 
    }

    shrLog("Comparing to the CPU results...\n\n");
    sumDelta = 0;
    sumRef   = 0;
    unsigned int distance = ((unsigned int)-1) / (QRNG_DIMENSIONS * N + 1);
    for(pos = 0; pos < QRNG_DIMENSIONS * N; pos++){
        unsigned int d = (pos + 1) * distance;
        ref       = MoroInvCNDcpu(d);
        delta     = (double)h_OutputGPU[pos] - ref;
        sumDelta += fabs(delta);
        sumRef   += fabs(ref);
    }
    L1norm = sumDelta / sumRef;
    shrLog("  L1 norm: %E\n", L1norm);
    shrLog("  ckInverseCNDGPU deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag &= (L1norm < 1e-6);

    // NOTE:  Most properly this should be done at any of the exit points above, but it is omitted elsewhere for clarity.
    shrLog("Release CPU buffers and OpenCL objects...\n\n"); 
    free(h_OutputGPU); 
    free(progSource);
    free(cdDevices);
    for (cl_uint i = 0; i < nDevice; i++)
    {
        clReleaseMemObject(d_Output[i]);
        clReleaseMemObject(c_Table[i]);
        clReleaseCommandQueue(cqCommandQueue[i]);
    }
    clReleaseKernel(ckQuasirandomGenerator);
    clReleaseKernel(ckInverseCNDGPU);
    clReleaseProgram(cpProgram);
    clReleaseContext(cxGPUContext);
    delete(szWorkgroup);

    // finish
    shrQAFinishExit(argc, (const char **)argv, bPassFlag ? QA_PASSED : QA_FAILED);

    shrEXIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper for OpenCL Niederreiter quasirandom number generator kernel
///////////////////////////////////////////////////////////////////////////////
void QuasirandomGeneratorGPU(cl_command_queue cqCommandQueue,
			     cl_kernel ckQuasirandomGenerator,
			     cl_mem d_Output,
			     cl_mem c_Table,
			     unsigned int seed,
			     unsigned int N,
                 size_t szWgXDim)
{
    cl_int ciErr;
    size_t globalWorkSize[2] = {shrRoundUp(szWgXDim, 128*128), QRNG_DIMENSIONS};
    size_t localWorkSize[2] = {szWgXDim, QRNG_DIMENSIONS};
    
    ciErr  = clSetKernelArg(ckQuasirandomGenerator, 0, sizeof(cl_mem),       (void*)&d_Output);
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 1, sizeof(cl_mem),       (void*)&c_Table );
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 2, sizeof(unsigned int), (void*)&seed    );
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 3, sizeof(unsigned int), (void*)&N       );
    ciErr |= clEnqueueNDRangeKernel(cqCommandQueue, ckQuasirandomGenerator, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);  
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper for OpenCL Inverse Cumulative Normal Distribution generator kernel
///////////////////////////////////////////////////////////////////////////////
void InverseCNDGPU(cl_command_queue cqCommandQueue, 
		   cl_kernel ckInverseCNDGPU, 
		   cl_mem d_Output, 
		   unsigned int pathN,
           unsigned int iDevice,
           unsigned int nDevice,
           size_t szWgXDim)
{
    cl_int ciErr;
    size_t globalWorkSize[1] = {shrRoundUp(szWgXDim, 128*128)};
    size_t localWorkSize[1] = {szWgXDim};

    ciErr  = clSetKernelArg(ckInverseCNDGPU, 0, sizeof(cl_mem),       (void*)&d_Output);
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 1, sizeof(unsigned int), (void*)&pathN   );
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 2, sizeof(unsigned int), (void*)&iDevice );
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 3, sizeof(unsigned int), (void*)&nDevice );
    ciErr |= clEnqueueNDRangeKernel(cqCommandQueue, ckInverseCNDGPU, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    oclCheckErrorEX(ciErr, CL_SUCCESS, NULL); 
}
