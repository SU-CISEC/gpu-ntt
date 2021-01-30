#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __forceinline__ void singleBarrett(unsigned long long& a, unsigned& q, unsigned& mu, int& qbit)  // ??
{  
    unsigned long long rx;
    rx = a >> (qbit - 2);
    rx *= mu;
    rx >>= qbit + 2;
    rx *= q;
    a -= rx;

    if (a >= q)
        a -= q;
}

__global__ void CTBasedNTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psi_powers[])
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned shared_array[];

#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying to shared memory from global
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * 2048];
    }

#pragma unroll
    for (int length = 1; length < 2048; length *= 2)
    {  // iterations for ntt
        int step = (2048 / length) / 2;
        int psi_step = local_tid / step;
        int target_index = psi_step * step * 2 + local_tid % step;;

        psi_step = (local_tid + blockIdx.x * 1024) / step;

        unsigned psi = psi_powers[length + psi_step];

        unsigned first_target_value = shared_array[target_index];
        unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        temp_storage *= psi;

        singleBarrett(temp_storage, q, mu, qbit);
        unsigned second_target_value = temp_storage;

        unsigned target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    }

}

__global__ void GSBasedINTTInnerSingle(unsigned a[], unsigned q, unsigned mu, int qbit, unsigned psiinv_powers[])
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned shared_array[];

    unsigned q2 = (q + 1) >> 1;

#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying to shared memory from global
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * 2048];
    }

    __syncthreads();

#pragma unroll
    for (int length = 1024; length >= 1; length /= 2)
    {  // iterations for intt
        int step = (2048 / length) / 2;

        int psi_step = local_tid / step;
        int target_index = psi_step * step * 2 + local_tid % step;

        psi_step = (local_tid + blockIdx.x * 1024) / step;

        unsigned psiinv = psiinv_powers[length + psi_step];

        unsigned first_target_value = shared_array[target_index];
        unsigned second_target_value = shared_array[target_index + step];

        unsigned target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

        first_target_value += q * (first_target_value < second_target_value);

        unsigned long long temp_storage = first_target_value - second_target_value;

        temp_storage *= psiinv;

        singleBarrett(temp_storage, q, mu, qbit);

        unsigned temp_storage_low = temp_storage;

        shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        

        __syncthreads();
    }

#pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    }
}