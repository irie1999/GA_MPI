/*
 * update_H_PML.cpp
 *
 *  Created on: 2021/06/10
 *      Author: ando
 */

#include <cmath>
#include "fdtd2d.h"

void update_Hr_PML(double **Hr, double **Hr1, double **Hr2,
    double **Eph, double *C1, double *C2){

  constexpr int J_OS { Nth - PML_L };

#pragma omp paralell for
  for(int i = 1; i < Nr; i++){
    for(int j = Nth - PML_L; j < Nth; j++){
      Hr1[i][j - J_OS] = C1[j - J_OS] * Hr1[i][j - J_OS] -
          C2[j - J_OS] / MU0 / r(i) / dth *
          ( Eph[i][j+1] - Eph[i][j] );

      Hr2[i][j - J_OS] = Hr2[i][j - J_OS] -
          Dt * std::cos( (j+0.5)*dth ) / std::sin( (j+0.5)*dth ) / 2.0 / MU0 / r(i)
          * ( Eph[i][j+1] + Eph[i][j] );

      Hr[i][j] = Hr1[i][j - J_OS] + Hr2[i][j - J_OS];
    }
  }
}

void update_Hph_PML(double **Hph, double **Hph_r, double **Hph_th,
    double **Er, double **Eth, double *C1, double *C2,
    double *Rs, double *Ls, double **Bph, double *Bph_r, double *Bph_th,
    const int NEW){

  constexpr int J_OS { Nth - PML_L };
  const int OLD { 1 - NEW };

  double c1 = r(1.0)*Dt/r(0.5)/dr;
#pragma omp paralell for
  for(int j = Nth - PML_L; j < Nth; j++){
    Bph_r[j - J_OS] = Bph_r[j - J_OS] - c1 * Eth[1][j];
    Bph_th[j - J_OS] = C1[j - J_OS] * Bph_th[j - J_OS] +
        C2[j - J_OS] / r(0.5) / dth *
        ( Er[0][j+1] - Er[0][j] );
    Bph[NEW][j - J_OS] = Bph_r[j - J_OS] + Bph_th[j - J_OS];

    double alph = r(0.0) * (Rs[j] + Rs[j+1]) / 2.0 / r(0.5) / dr;
    double beta = r(0.0) * (Ls[j] + Ls[j+1]) / 2.0 / r(0.5) / dr + MU0;
    double ch1 = (beta/Dt - alph/2.0 ) / (beta/Dt + alph/2.0 );
    double ch2 = 1.0 / (beta/Dt + alph/2.0 ) / Dt;
    Hph[0][j] = ch1 * Hph[0][j] + ch2 * ( Bph[NEW][j - J_OS] - Bph[OLD][j - J_OS] );
  }

#pragma omp paralell for
  for(int i = 1; i < Nr; i++){
    for(int j = Nth - PML_L; j < Nth; j++){
      Hph_r[i][j - J_OS] = Hph_r[i][j - J_OS] -
          Dt / MU0 / r(i+0.5) / dr *
          ( r(i+1.0) * Eth[i+1][j] - r(i) * Eth[i][j] );

      Hph_th[i][j - J_OS] = C1[j - J_OS] * Hph_th[i][j - J_OS] +
          C2[j - J_OS] / MU0 / r(i+0.5) / dth *
          ( Er[i][j+1] - Er[i][j] );

      Hph[i][j] = Hph_r[i][j - J_OS] + Hph_th[i][j - J_OS];
    }
  }
}

