// coded by Can Elgezen and Özgün Özerk
// contributed by Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk

#pragma once
#include <stdlib.h>
#include <random>

unsigned modpow64(unsigned a, unsigned b, unsigned mod)  // calculates (<a> ** <b>) mod <mod>
{
    unsigned res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        unsigned long long t64 = (unsigned long long)a * a;
        a = t64 % mod;
        if (b & 1)
        {
            unsigned long long r64 = (unsigned long long)a * res;
            res = r64 % mod;
        }

    }
    return res;
}

unsigned long long bitReverse(unsigned long long a, int bit_length)  // reverses the bits for twiddle factor calculation
{
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

std::random_device dev;  // uniformly distributed integer random number generator that produces non-deterministic random numbers
std::mt19937_64 rng(dev());  // pseudo-random generator of 64 bits with a state size of 19937 bits

void randomArray64(unsigned a[], int n, unsigned q)
{
    std::uniform_int_distribution<unsigned> randnum(0, q - 1);  // uniformly distributed random integers on the closed interval [a, b] according to discrete probability

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void fillTablePsi64(unsigned psi, unsigned q, unsigned psiinv, unsigned psiTable[], unsigned psiinvTable[], unsigned int n)  // twiddle factors computation
{
    for (int i = 0; i < n; i++)
    {
        psiTable[i] = modpow64(psi, bitReverse(i, log2(n)), q);
        psiinvTable[i] = modpow64(psiinv, bitReverse(i, log2(n)), q);
    }
}

