# gpu-ntt
Number Theoretic Transform Implementation on GPU for FHE Applications

--------------------

## File Descriptions:

### helper.h
includes:

- modular power calculation
- reversing bits
- random array creation
- calculating twiddle factors


### ntt_30bit.cuh
includes gpu functions:

- barrett
- ntt
- intt

### 30bit_ntt_test.cu
includes:

the main program for applying ntt, then intt on a randomly generated array


--------------------

## How to run

Compile with: *nvcc -arch=sm_XX -rdc=true -cudart static --machine 64 -use_fast_math -O2 30bit_ntt_test.cu -o 30bit_ntt -lcudadevrt -std=c++11*

Run with: *./30bit_ntt*


--------------------


*Coded by: Can Elgezen, Özgün Özerk* \
*Contributed by: Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk*
