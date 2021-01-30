#include <iostream>
using std::cout;  // yes we are that lazy
using std::endl;  // :)

#include "ntt_30bit.cuh"
#include "helper.h"

#define check 1

int main()
{
    unsigned n = 2048;

    int size_array = sizeof(unsigned) * n;
    int size = sizeof(unsigned);

    unsigned q = 536608769, psi = 284166, psiinv = 208001377, ninv = 536346753;  // parameter initialization
    unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    unsigned* psiTable = (unsigned*)malloc(size_array);
    unsigned* psiinvTable = (unsigned*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned* psi_powers, * psiinv_powers;

    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    cout << "n = " << n << endl;
    cout << "q = " << q << endl;
    cout << "Psi = " << psi << endl;
    cout << "Psi Inverse = " << psiinv << endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    double mu1 = powl(2, 2 * bit_length);
    unsigned mu = mu1 / q;

    unsigned* a;
    cudaMallocHost(&a, sizeof(unsigned) * n);
    randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1

    unsigned* res_a;
    cudaMallocHost(&res_a, sizeof(unsigned) * n);

    unsigned* d_a;
    cudaMalloc(&d_a, size_array);

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, 0);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned), 0>>>(d_a, q, mu, bit_length, psi_powers);
    GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpyAsync(res_a, d_a, size_array, cudaMemcpyDeviceToHost, 0);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    bool correct = 1;
    if (check) //check the correctness of results
    {
        for (int i = 0; i < n; i++)
        {
            if (a[i] != res_a[i])
            {
                correct = 0;
                break;
            }
        }
    }

    if (correct)
        cout << "\nNTT and INTT are working correctly." << endl;
    else
        cout << "\nNTT and INTT are not working correctly." << endl;

    cudaFreeHost(a); cudaFreeHost(res_a);  
    cudaFree(d_a);
    return 0;
}


