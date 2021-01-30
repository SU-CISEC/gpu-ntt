# gpu-ntt
Number Theoretic Transform Implementation on GPU for FHE Applications



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


*Coded by: Can Elgezen, Özgün Özerk* //
*Contributed by: Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk*
