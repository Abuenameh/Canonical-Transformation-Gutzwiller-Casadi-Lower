/* 
 * File:   mathematica.hpp
 * Author: Abuenameh
 *
 * Created on 15 November 2013, 16:20
 */

#ifndef MATHEMATICA_HPP
#define	MATHEMATICA_HPP


#include <iostream>
#include <iomanip>
#include <deque>
#include <vector>
#include <map>
#include <string>
#include <complex>

using std::cout;
using std::string;
using std::pair;
using std::deque;
using std::ostream;
using std::ostringstream;
using std::numeric_limits;
using std::setprecision;
using std::complex;
using std::vector;

#include <boost/multi_array.hpp>
#include <boost/algorithm/string.hpp>

using boost::multi_array;

using namespace boost::algorithm;

//#include <matrix.h>
//
//using namespace itensor;

template<typename T>
class mathematica {
public:

    mathematica(T& v_) : v(v_) {
    }

    T& v;
};

template<>
class mathematica<int> {
public:

    mathematica(int i_) : i(i_) {
    }

    int i;
};

template<>
class mathematica<const int> {
public:

    mathematica(int i_) : i(i_) {
    }

    int i;
};

template<>
class mathematica<double> {
public:

    mathematica(double d_) : d(d_) {
    }

    double d;
};

template<>
class mathematica<bool> {
public:

    mathematica(bool b_) : b(b_) {
    }

    bool b;
};

template<>
class mathematica<complex<double> > {
public:

    mathematica(complex<double> c_) : c(c_) {
    }

    complex<double> c;
};

ostream& operator<<(ostream& out, const mathematica<int> m) {
    out << m.i;
    return out;
}

ostream& operator<<(ostream& out, const mathematica<const int> m) {
    out << m.i;
    return out;
}

ostream& operator<<(ostream& out, const mathematica<double> m) {
    double d = m.d;
    ostringstream oss;
    oss << setprecision(numeric_limits<double>::digits10) << d;
    out << replace_all_copy(oss.str(), "e", "*^");
    return out;
}

ostream& operator<<(ostream& out, const mathematica<bool> m) {
    out << (m.b ? "True" : "False");
    return out;
}

ostream& operator<<(ostream& out, const mathematica<complex<double> > m) {
    complex<double> c = m.c;
    out << "(" << mathematica<double>(c.real()) << ")+I(" << mathematica<double>(c.imag()) << ")";
    return out;
}

ostream& operator<<(ostream& out, const mathematica<string> m) {
    out << "\"" << m.v << "\"";
    return out;
}

//ostream& operator<<(ostream& out, const mathematica<Vector>& m) {
//    VectorRef& v = m.v;
//    int len = v.Length();
//    out << "{" << mathematica<Real>(v(1));
//    for (int i = 2; i <= len; i++) {
//        out << "," << mathematica<Real>(v(i));
//    }
//    out << "}";
//    return out;
//}
//
//ostream& operator<<(ostream& out, const mathematica<Matrix>& m) {
//    MatrixRef& mat = m.v;
//    int r = mat.Nrows();
//    int c = mat.Ncols();
//    out << "{{" << mathematica<Real> (mat(1,1));
//    for (int j = 2; j <= c; j++) {
//        out << "," << mathematica<Real> (mat(1,j));
//    }
//    out << "}";
//    for (int i = 2; i <= r; i++) {
//        out << ",{" << mathematica<Real> (mat(i,1));
//        for (int j = 2; j <= c; j++) {
//            out << "," << mathematica<Real> (mat(i,j));
//        }
//        out << "}";
//    }
//    out << "}";
//    return out;
//}

template<typename T, typename Alloc>
ostream& operator<<(ostream& out, const mathematica<deque<T, Alloc> >& m) {
    deque<T, Alloc>& d = m.v;
    out << "{" << mathematica<T > (d[0]);
    for (int i = 1; i < d.size(); i++) {
        out << "," << mathematica<T > (d[i]);
    }
    out << "}";
    return out;
}

template<typename T, typename Alloc>
ostream& operator<<(ostream& out, const mathematica<vector<T, Alloc> >& m) {
    vector<T, Alloc>& d = m.v;
    out << "{" << mathematica<T > (d[0]);
    for (int i = 1; i < d.size(); i++) {
        out << "," << mathematica<T > (d[i]);
    }
    out << "}";
    return out;
}

template<typename T, typename Alloc>
ostream& operator<<(ostream& out, const mathematica<multi_array<T, 2, Alloc > >& m) {
    multi_array<T, 2, Alloc> & ma = m.v;
    int r = ma.shape()[0];
    int c = ma.shape()[1];
    out << "{{" << mathematica<T > (ma[0][0]);
    for (int j = 1; j < c; j++) {
        out << "," << mathematica<T > (ma[0][j]);
    }
    out << "}";
    for (int i = 1; i < r; i++) {
        out << ",{" << mathematica<T > (ma[i][0]);
        for (int j = 1; j < c; j++) {
            out << "," << mathematica<T > (ma[i][j]);
        }
        out << "}";
    }
    out << "}";
    return out;
}

template<typename T, typename Alloc>
ostream& operator<<(ostream& out, const mathematica<multi_array<T, 3, Alloc > >& m) {
    multi_array<T, 3, Alloc> & ma = m.v;
    int r = ma.shape()[0];
    int c = ma.shape()[1];
    int d = ma.shape()[2];
    out << "{{{" << mathematica<T > (ma[0][0][0]);
    for (int k = 1; k < d; k++) {
        out << "," << mathematica<T > (ma[0][0][k]);
    }
    out << "}";
    for (int j = 1; j < c; j++) {
        out << ",{" << mathematica<T > (ma[0][j][0]);
        for (int k = 1; k < d; k++) {
            out << "," << mathematica<T > (ma[0][j][k]);
        }
        out << "}";
    }
    out << "}";
    for (int i = 1; i < r; i++) {
        out << ",{{" << mathematica<T > (ma[i][0][0]);
        for (int k = 1; k < d; k++) {
            out << "," << mathematica<T > (ma[i][0][k]);
        }
        out << "}";
        for (int j = 1; j < c; j++) {
            out << ",{" << mathematica<T > (ma[i][j][0]);
            for (int k = 1; k < d; k++) {
                out << "," << mathematica<T > (ma[i][j][k]);
            }
            out << "}";
        }
        out << "}";
    }
    out << "}";
    return out;

}

template<typename T, typename U>
ostream& operator<<(ostream& out, const mathematica<pair<T, U> >& p) {
    out << "{" << mathematica<T>(p.v.first) << "," << mathematica<U>(p.v.second) << "}";
    return out;
}

template<typename T, size_t N>
ostream& operator<<(ostream& out, const mathematica<boost::array<T, N> >& arr) {
    boost::array<T, N>& a = arr.v;
    out << "{" << mathematica<T>(a[0]);
    for (int i = 1; i < N; i++) {
        out << "," << mathematica<T>(a[i]);
    }
    out << "}";
    return out;
}

template<typename T>
mathematica<T> math(T& t) {
    return mathematica<T > (t);
}

mathematica<double> math(double d) {
    return mathematica<double>(d);
}

mathematica<complex<double> > math(complex<double> c) {
    return mathematica<complex<double> >(c);
}


template<typename T> void printMath(ostream& out, string name, T& t) {
    out << name << "=" << ::math(t) << ";" << std::endl;
}

template<typename T> void printMath(ostream& out, string name, int i, T& t) {
    out << name << "[" << i << "]" << "=" << ::math(t) << ";" << std::endl;
}



#endif	/* MATHEMATICA_HPP */

