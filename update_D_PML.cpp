/*
 * update_D_PML.cpp
 *
 *  Created on: 2021/06/10
 *      Author: ando
 */
#include "fdtd2d.h"

void update_Dr_PML(double **Dr, double **Dr1, double **Dr2,
    double **Hph, double *C1, double *C2){

  constexpr int J_OS { Nth - PML_L }; /* Offset for sigma */

#pragma omp paralell for
  for(int i = 0; i < Nr; i++){
    for(int j = Nth - PML_L; j < Nth; j++){
      Dr1[i][j - J_OS] = C1[j - J_OS] * Dr1[i][j - J_OS] +
          C2[j - J_OS] / r(i+0.5) / dth *
          ( Hph[i][j] - Hph[i][j-1] );

      Dr2[i][j - J_OS] = Dr2[i][j - J_OS] +
          Dt * std::cos(j*dth) / std::sin(j*dth) / 2.0 / r(i+0.5) *
          ( Hph[i][j] + Hph[i][j-1] );

      Dr[i][j] = Dr1[i][j - J_OS] + Dr2[i][j - J_OS];
    }
  }
}

void update_Dph_PML(double **Dph, double **Dph_r, double **Dph_th,
    double **Hth, double **Hr,
    double *C1, double *C2){

  constexpr int J_OS { Nth - PML_L }; /* Offset for sigma */

#pragma omp paralell for
  for(int i = 1; i < Nr; i++){
    for(int j = Nth - PML_L; j < Nth; j++){
      Dph_r[i][j - J_OS] = Dph_r[i][j - J_OS] +
          Dt / r(i) / dr *
          ( r(i+0.5) * Hth[i][j] - r(i-0.5) * Hth[i-1][j] );

      Dph_th[i][j - J_OS] = C1[j - J_OS] * Dph_th[i][j - J_OS] -
          C2[j - J_OS] / r(i) / dth *
          ( Hr[i][j] - Hr[i][j-1] );

      Dph[i][j] = Dph_r[i][j - J_OS] + Dph_th[i][j - J_OS];
    }
  }
}
