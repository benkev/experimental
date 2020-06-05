/*
 * gpu_rand_unif64.c
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <locale.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "vanvleck_structs.h"
#include "simulate_vvcorr.h"
#ifndef __OPENCL_VERSION__
#    define __OPENCL_VERSION__ 0
#endif

/* cl_ulong const Nwitem = 128, Nwgroup = 10; */
/* cl_uint const nrnd=10000; */
cl_ulong Nwitem = 256, Nwgroup = 100;
cl_uint nrnd=10000;


int main(int argc, char *argv[]) {

    setlocale(LC_NUMERIC, "");

    if (argc == 4) {
        nrnd = (cl_ulong) atoi(argv[1]);
        Nwgroup = (cl_ulong) atoi(argv[2]);
        Nwitem = (cl_ulong) atoi(argv[3]);
    }
    else {
        fprintf(stderr, "Error: wrong number of parameters!\n\n");
        fprintf(stderr, "Run %s with 3 parameters:\n\n" 
                "%s  nrnd  Nwgroup  Nwitem\n\nwhere\n"
                "nrnd:    # of random numbers to generate in one thread;\n"
                "Nwgroup: # of OpenCL work groups/CUDA blocks;\n"
                "Nwitem:  # of OpenCL work items/CUDA threads per block;\n",
                argv[0], argv[0]);
                fprintf(stderr, "Example:\n\n"
                "%s  10000  50  256\n\n"
                "Note: The total size of device memory required is "
                "nrnd * Nwitem * Nwgroup * 8 / (1024^3) GiB,\n"
                "       it cannot exceed the maximum allocation size for the "
                "particular GPU.\n\n", argv[0]);
        exit(1);
    }
    printf("Requires %.2f GiB on GPU.\n\n", 8.*(double)(nrnd*Nwitem*Nwgroup) / 
           (1024.*1024.*1024.));

    cl_uint i;
    FILE *fp;
    char *source_str=NULL;
    size_t source_size;
    fp = fopen("ker_rand_unif64.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file.\n"); exit(1);
    }

    cl_ulong nrnd_total = Nwitem * Nwgroup * nrnd;
    cl_double *rndu = (cl_double *) malloc(nrnd_total*sizeof(cl_double));
    if (rndu == NULL) {
        printf("Error! rndu[nrnd_total] not allocated.");
        exit(0);
    }
    
    /*
     * Find the kernel file size source_size, allocate memory for it, 
     * and read it in the source_str[source_size] string.
     */
    fseek(fp, 0L, SEEK_END);
    source_size = ftell(fp);
    rewind(fp); /* Reset the file pointer to 'fread' from the beginning */
    source_str = (char*) malloc((source_size + 1)*sizeof(char));
    fread( source_str, sizeof(char), source_size, fp);
    source_str[source_size] = '\0';
    fclose( fp );
    
    /*
     * Initialize random states
     */

    /* Number of parallel random number sequences */
    cl_ulong Nproc = Nwitem*Nwgroup;
    cl_ulong seed = 90752;
    rand64State *rndst = (rand64State *) malloc(Nproc*sizeof(rand64State));
    if (rndst == NULL) {
        printf("Error! rndst[Nproc] not allocated.");
        exit(0);
    }
    rand64_initn(seed, Nproc, rndst);

    cl_ulong memst = Nproc*sizeof(rand64State);
    cl_ulong memrnd = nrnd_total*sizeof(cl_ulong);
    printf("Work items: %'ld\n", Nwitem);
    printf("Work groups: %'ld\n", Nwgroup);
    printf("Processes (treads) total : %'ld\n", Nproc);
    printf("Numbers in one thread nrnd: %'ld\n", nrnd);
    printf("Total numbers nrnd_total: %'ld\n", nrnd_total);
    printf("Memory for %'ld PRNG states: %'ld B = %'.1f KiB = %'.1f MiB\n", \
           Nproc, memst, memst/1024., memst/1024./1024.);
    printf("Memory for %'ld 64-bit random numbers: " \
           "%'ld B = %'.1f KiB = %'.1f MiB = %'.2f GiB\n",
           nrnd_total, memrnd, memrnd/1024., memrnd/1024./1024.,    \
           memrnd/1024./1024./1024.);

    /*
     * Get platform and device information
     */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    
    // ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    printf("ret_num_platforms=%u, ret=%u\n", ret_num_platforms, ret);
    
    // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, 
    //        NULL, &ret_num_devices);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, 
                         &device_id, &ret_num_devices);

    printf("ret_num_devices=%u, ret=%u\n", ret_num_devices, ret);
    
    /* Create an OpenCL context: default properties, for 1 device, no callback,
     * no user_data */
    cl_context context =                                            \
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    
    /* Create a command queue for one device */
    cl_command_queue cmd_queue = \
        clCreateCommandQueue(context, device_id, 0, &ret);

    /* 
     * Create memory buffers ON THE DEVICE for the PRNG states and the
     * random number array created by the kernel  
     */
    cl_mem rndst_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                          Nproc*sizeof(rand64State), NULL, &ret);    
    if (ret) printf("clCreateBuffer rndst: ret=%d\n", ret);

    cl_mem rndu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                          nrnd_total*sizeof(cl_double), NULL, &ret);
    if (ret) printf("clCreateBuffer rndu:  ret=%d\n", ret);

    /* Copy the PRNG states to their respective memory buffer on device */
    ret = clEnqueueWriteBuffer(cmd_queue, rndst_mem, CL_TRUE, 0,
            Nproc*sizeof(rand64State), rndst, 0, NULL, NULL);
    if (ret) printf("clEnqueueWriteBuffer ret=%d\n", ret);

    /* 
     * Create a program from the kernel source 
     */
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    if (ret) printf("clCreateProgramWithSource ret=%d\n", ret);

    /* Build the program */
    cl_char *opts = "-I .";

    ret = clBuildProgram(program, 1, &device_id, opts, NULL, NULL);

    if (ret) printf("clBuildProgram ret=%d\n", ret);

    /*
     * Print the CL compiler output, if errors occurred
     */
    if(ret < 0) {
        size_t log_size;
        char *program_log=NULL;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, \
                              0, NULL, &log_size);
        program_log = (char *) calloc(log_size+1, sizeof(char));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, \
                              log_size+1, program_log, NULL);
        printf("%s\n", program_log);
        printf("Log size: %lu bytes.\n", log_size);
        free(program_log);
    }


    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "genrand", &ret);

    printf("clCreateKernel ret=%d\n", ret);

    /* Set the arguments of the kernel */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&rndst_mem);
    if (ret) printf("clSetKernelArg rndst: ret=%d\n", ret);
    
    ret = clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&nrnd);
    if (ret) printf("clSetKernelArg rndu: ret=%d\n", ret);
    
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&rndu_mem);
    if (ret) printf("clSetKernelArg rndu: ret=%d\n", ret);
    
    /* 
     * Execute the OpenCL kernel
     */
    ret = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, 
            &Nproc, &Nwitem, 0, NULL, NULL);
    if (ret) printf("clEnqueueNDRangeKernel: ret=%d\n", ret);

    /* 
     * Read the memory buffer rndu on the device to the local variable rndu
     */
    ret = clEnqueueReadBuffer(cmd_queue, rndu_mem, CL_TRUE, 0, \
                              nrnd_total*sizeof(cl_ulong), rndu, 0, NULL, NULL);
    if (ret) printf("clEnqueueReadBuffer rndu: ret=%d\n", ret);

    printf("Application rndst:\n");
    for (int i=0; i<4; i++)
        printf("%20ld %20ld %20ld %20ld\n", rndst[0].s[i], rndst[1].s[i], \
               rndst[2].s[i], rndst[3].s[i]);


    /*
     * rndu_local points into rndu[nrnd_total] treating it as rndu[Nproc][nrnd].
     */
    cl_double *rndu_local;
    cl_double rnd, rndmin = 1e20, rndmax = 1e-20;
    cl_double *rndu_avg = (cl_double *) calloc(Nproc, sizeof(cl_double));
    
    for (size_t iseq=0; iseq<Nproc; iseq++) {
        rndu_local = rndu + iseq*nrnd;
        for (cl_ulong irnd=0; irnd<nrnd; irnd++) {
            rnd = *rndu_local;
            rndu_avg[iseq] += rnd;
            if (rnd < rndmin) rndmin = rnd;
            if (rnd > rndmax) rndmax = rnd;
            rndu_local++;
        }
        rndu_avg[iseq] /= (cl_double) nrnd; /* Save average over nrand */
    }
    
    printf("Application rndu:\n");
    for (int i=265; i<301; i++)
        printf("%20.16f %20.16f %20.16f %20.16f \n", rndu[i], rndu[nrnd+i], \
               rndu[2*nrnd+i], rndu[3*nrnd+i]);

    printf("Application rndu averages:\n");
    printf("%20.16f %20.16f %20.16f %20.16f \n", rndu_avg[0], rndu_avg[1], \
               rndu_avg[2], rndu_avg[3]);

    printf("\n");
    printf("Application rndmin = %20.16f rndmax = %20.16f\n", rndmin, rndmax);


    /* Clean up*/
    ret = clFlush(cmd_queue);
    ret = clFinish(cmd_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(rndu_mem);
    ret = clReleaseMemObject(rndst_mem);
    ret = clReleaseCommandQueue(cmd_queue);
    ret = clReleaseContext(context);

    /*
     * The predefined macro symbol __OPENCL_VERSION__ can be used 
     * to determine if we are in a kernel code or in the application,
     * to expand into different code (like data types in structures).
     */
    
    return 0;
}












