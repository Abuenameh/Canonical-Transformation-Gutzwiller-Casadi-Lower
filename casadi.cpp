#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

using namespace boost;
using namespace boost::filesystem;

#include "casadi.hpp"

double energyfunc(const vector<double>& x, vector<double>& grad, void *data) {
    GroundStateProblem* prob = static_cast<GroundStateProblem*> (data);
    return prob->E(x, grad);
}

void compileFunction(Function& f, string name) {
    string sourcename = name + ".c";
    f.generateCode(sourcename);
    string libraryname = name + ".so";
    string cmd;
#ifdef AMAZON
    cmd = "gcc -shared -fPIC -O2 -o " + libraryname + " " + sourcename;
#else
    cmd = "gcc -shared -O2 -o " + libraryname + " " + sourcename;
#endif
    ::system(cmd.c_str());
}

void loadFunction(Function& f, string name) {
    string libraryname = name + ".so";
    f = ExternalFunction(libraryname);
    f.init();
}

namespace casadi {

    inline bool isnan(SX& sx) {
        return sx.at(0).isNan();
    }

    inline bool isinf(SX sx) {
        return sx.at(0).isInf();
    }
}

void GroundStateProblem::setup() {

    string compiledir = "L" + to_string(L) + "nmax" + to_string(nmax);
    path compilepath(compiledir);

    if (!exists(compilepath)) {

        create_directory(compilepath);

        vector<SX> fin = SX::sym("f", 1, 1, 2 * L * dim);
        vector<SX> dU = SX::sym("dU", 1, 1, L);
        vector<SX> J = SX::sym("J", 1, 1, L);
        SX U0 = SX::sym("U0");
        SX mu = SX::sym("mu");
        SX theta = SX::sym("theta");

        vector<SX> params;
        params.push_back(U0);
        for (SX sx : dU) params.push_back(sx);
        for (SX sx : J) params.push_back(sx);
        params.push_back(mu);
        params.push_back(theta);

        SX x = SX::sym("x", fin.size());
        SX p = SX::sym("p", params.size());

        vector<SX> xs;
        for (int i = 0; i < x.size(); i++) {
            xs.push_back(x.at(i));
        }
        vector<SX> ps;
        for (int i = 0; i < p.size(); i++) {
            ps.push_back(p.at(i));
        }

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
            SX E = energy(i, n, fin, J, U0, dU, mu, theta);
            E = substitute(vector<SX>{E}, fin, xs)[0];
            E = substitute(vector<SX>{E}, params, ps)[0];

            SXFunction Ef = SXFunction(nlpIn("x", x, "p", p), nlpOut("f", E));
            Ef.init();
            compileFunction(Ef, compiledir + "/E." + to_string(i) + "." + to_string(n));
            Function Egradf = Ef.gradient(NL_X, NL_F);
            Egradf.init();
            compileFunction(Egradf, compiledir + "/Egrad." + to_string(i) + "." + to_string(n));
            }
        }

    }
}

//boost::mutex problem_mutex;

GroundStateProblem::GroundStateProblem() {
    string compiledir = "L" + to_string(L) + "nmax" + to_string(nmax);
    Ef = vector<Function>(L*(nmax+1));
    Egradf = vector<Function>(L*(nmax+1));
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            int j = i*(nmax+1)+n;
        loadFunction(Ef[j], compiledir + "/E." + to_string(i) + "." + to_string(n));
        loadFunction(Egradf[j], compiledir + "/Egrad." + to_string(i) + "." + to_string(n));
        }
    }
}

string GroundStateProblem::getRuntime() {
    time_period period(start_time, stop_time);
    return to_simple_string(period.length());
}

double GroundStateProblem::E(const vector<double>& f, vector<double>& grad) {
    double E = 0;
    for (int i = 0; i < Ef.size(); i++) {
        double Ei = 0;
        Ef[i].setInput(f.data(), NL_X);
        Ef[i].setInput(params.data(), NL_P);
        Ef[i].evaluate();
        Ef[i].getOutput(Ei, NL_F);
        E += Ei;
    }
    if (!grad.empty()) {
        fill(grad.begin(), grad.end(), 0);
        for (int i = 0; i < Egradf.size(); i++) {
            vector<double> gradi(2 * L * dim);
            Egradf[i].setInput(f.data(), NL_X);
            Egradf[i].setInput(params.data(), NL_P);
            Egradf[i].evaluate();
            Egradf[i].output().getArray(gradi.data(), gradi.size(), DENSE);
            for (int j = 0; j < 2 * L * dim; j++) {
                grad[j] += gradi[j];
            }
        }
    }
    return E;
}

void GroundStateProblem::setParameters(double U0, vector<double>& dU, vector<double>& J, double mu) {
    params.clear();
    params.push_back(U0);
    params.insert(params.end(), dU.begin(), dU.end());
    params.insert(params.end(), J.begin(), J.end());
    params.push_back(mu);
    params.push_back(0);
}

void GroundStateProblem::setTheta(double theta) {
    params.back() = theta;
}

SX GroundStateProblem::energy(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {

    SX E = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
        E += energy(i, n, fin, J, U0, dU, mu, theta);
        }
    }
    return E;
}

SX GroundStateProblem::energy(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta) {

    complex<SX> expth = complex<SX>(cos(theta), sin(theta));
    complex<SX> expmth = ~expth;
    complex<SX> exp2th = expth*expth;
    complex<SX> expm2th = ~exp2th;

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        f[j] = reinterpret_cast<complex<SX>*> (&fin[2 * j * dim]);
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    //    for (int i = 0; i < L; i++) {

    int k1 = mod(i - 2);
    int j1 = mod(i - 1);
    int j2 = mod(i + 1);
    int k2 = mod(i + 2);

    Ei = complex<SX>(0, 0);
    Ej1 = complex<SX>(0, 0);
    Ej2 = complex<SX>(0, 0);
    Ej1j2 = complex<SX>(0, 0);
    Ej1k1 = complex<SX>(0, 0);
    Ej2k2 = complex<SX>(0, 0);

//    for (int n = 0; n <= nmax; n++) {
        Ei += (0.5 * (U0 + dU[i]) * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

        if (n < nmax) {
            Ej1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                    * f[i][n] * f[j1][n + 1];
            Ej2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                    * f[j2][n + 1];

            if (n > 0) {
                Ej1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                        * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                Ej2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                        * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
            }
            if (n < nmax - 1) {
                Ej1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                        * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                Ej2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                        * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
            }

            if (n > 1) {
                Ej1 += -J[j1] * J[j1] * exp2th * g(n, n - 1) * g(n - 1, n)
                        * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                        * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                Ej2 += -J[i] * J[i] * expm2th * g(n, n - 1) * g(n - 1, n)
                        * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                        * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
            }
            if (n < nmax - 2) {
                Ej1 -= -J[j1] * J[j1] * exp2th * g(n, n + 3) * g(n + 1, n + 2)
                        * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                        * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                Ej2 -= -J[i] * J[i] * expm2th * g(n, n + 3) * g(n + 1, n + 2)
                        * (eps(dU, i, j2, n, n + 3, i, j2, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                        * ~f[i][n + 2] * ~f[j2][n + 1] * f[i][n] * f[j2][n + 3];
            }

            for (int m = 1; m <= nmax; m++) {
                if (n != m - 1) {
                    Ej1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                            ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                    Ej2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                            ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                    Ej1 += J[j1] * expth * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                            * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                    Ej2 += J[i] * expmth * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                            * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                    if (n != m - 3 && m > 1 && n < nmax - 1) {
                        Ej1 += -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                        Ej2 += -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                    }
                    if (n != m + 1 && n > 0 && m < nmax) {
                        Ej1 -= -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                        Ej2 -= -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                    }

                    if (n > 0) {
                        Ej1j2 += -J[j1] * J[i] * g(n, m) * g(n - 1, n)
                                * (eps(dU, i, j1, n, m, i, j2, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][n - 1]
                                * f[i][n - 1] * f[j1][m] * f[j2][n];
                        Ej1j2 += -J[i] * J[j1] * g(n, m) * g(n - 1, n)
                                * (eps(dU, i, j2, n, m, i, j1, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][n - 1]
                                * f[i][n - 1] * f[j2][m] * f[j1][n];
                    }
                    if (n < nmax - 1) {
                        Ej1j2 -= -J[j1] * J[i] * g(n, m) * g(n + 1, n + 2)
                                * (eps(dU, i, j1, n, m, i, j2, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][n + 1]
                                * f[i][n] * f[j1][m] * f[j2][n + 2];
                        Ej1j2 -= -J[i] * J[j1] * g(n, m) * g(n + 1, n + 2)
                                * (eps(dU, i, j2, n, m, i, j1, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][n + 1]
                                * f[i][n] * f[j2][m] * f[j1][n + 2];
                    }

                    Ej1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1)
                            * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                            * (~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m] -
                            ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]);
                    Ej2 += -0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1)
                            * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                            * (~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m] -
                            ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]);

                    for (int q = 1; q <= nmax; q++) {
                        if (n < nmax - 1 && n != q - 2) {
                            Ej1j2 += -0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                    * f[i][n] * f[j1][m] * f[j2][q];
                            Ej1j2 += -0.5 * J[i] * J[j1] * g(n, m) * g(n + 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                    * f[i][n] * f[j2][m] * f[j1][q];
                        }
                        if (n > 0 && n != q) {
                            Ej1j2 -= -0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                    * f[i][n - 1] * f[j1][m] * f[j2][q];
                            Ej1j2 -= -0.5 * J[i] * J[j1] * g(n, m) * g(n - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                    * f[i][n - 1] * f[j2][m] * f[j1][q];
                        }

                        if (m != q) {
                            Ej1k1 += -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                    * f[i][n] * f[j1][m] * f[k1][q];
                            Ej2k2 += -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                    * f[i][n] * f[j2][m] * f[k2][q];
                            Ej1k1 -= -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                    * f[i][n] * f[j1][m - 1] * f[k1][q];
                            Ej2k2 -= -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][q - 1]
                                    * f[i][n] * f[j2][m - 1] * f[k2][q];
                        }

                    }

                    for (int p = 0; p < nmax; p++) {

                        if (p != n - 1 && 2 * n - m == p && n > 0) {
                            Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                        }
                        if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                            Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n] * f[j2][m] * f[j1][p + 1];
                        }

                        if (p != n - 1 && 2 * n - m != p && n > 0) {
                            Ej1j2 += -0.25 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1)
                                    * (eps(dU, i, j1, n, m, i, j2, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 += -0.25 * J[i] * J[j1] * g(n, m) * g(n - 1, p + 1)
                                    * (eps(dU, i, j2, n, m, i, j1, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                        }
                        if (p != n + 1 && 2 * n - m != p - 2 && n < nmax - 1) {
                            Ej1j2 -= -0.25 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1)
                                    * (eps(dU, i, j1, n, m, i, j2, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n] * f[j1][m] * f[j2][p + 1];
                            Ej1j2 -= -0.25 * J[i] * J[j1] * g(n, m) * g(n + 1, p + 1)
                                    * (eps(dU, i, j2, n, m, i, j1, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n] * f[j2][m] * f[j1][p + 1];
                        }

                        if (p != m - 1 && n != p) {
                            Ej1k1 += -0.25 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, p + 1)
                                    * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                    * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                    ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                            Ej2k2 += -0.25 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, p + 1)
                                    * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                    * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                    ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                        }
                    }

                    Ej1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                            ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                            * f[i][n] * f[j1][m] * f[k1][n + 1]);
                    Ej2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                            ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                            * f[i][n] * f[j2][m] * f[k2][n + 1]);

                    Ej1k1 += -J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, m)
                            * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                            ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                    Ej2k2 += -J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, m)
                            * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                            ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                    if (m != n - 1 && n != m && m < nmax && n > 0) {
                        Ej1 += -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                        Ej2 += -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                    }
                    if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                        Ej1 -= -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                        Ej2 -= -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                    }
                }
            }
        }
//    }

    Ei /= norm2[i];
    Ej1 /= norm2[i] * norm2[j1];
    Ej2 /= norm2[i] * norm2[j2];
    Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
    Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
    Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

    E += Ei;
    E += Ej1;
    E += Ej2;
    E += Ej1j2;
    E += Ej1k1;
    E += Ej2k2;
    //    }

    return E.real();
}