/* 
 * File:   gutzwiller.hpp
 * Author: Abuenameh
 *
 * Created on August 10, 2014, 10:45 PM
 */

#ifndef GUTZWILLER_HPP
#define	GUTZWILLER_HPP

#include <complex>
#include <vector>
#include <iostream>

using namespace std;

const int L = 25;
const int nmax = 6;
const int dim = nmax + 1;

template<class T>
complex<T> operator~(const complex<T> a) {
	return conj(a);
}

struct Parameters {
    double theta;
};

inline int mod(int i) {
	return (i + L) % L;
}

inline SX g(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline SX eps(vector<SX>& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

inline SX eps(SX& U, int n, int m) {
    return (n - m + 1) * U;
}

inline SX eps(vector<SX>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n*U[i] - (m-1)*U[j] + (q-1)*U[k] - p*U[l];
}


#endif	/* GUTZWILLER_HPP */

