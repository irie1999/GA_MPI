/*
 * initialize_pml.cpp
 *
 *  Created on: 2021/06/10
 *      Author: ando
 */
#include "fdtd2d.h"

void initialize_pml(double *C01, double *C02, double *C11, double *C12){

  const double sigma_max = - (PML_M + 1.0) * C0 / 2.0 / PML_L / R0 / dth
      * std::log(Gamma);

  for(int i = 0; i <= PML_L; i++){
    double sigma0 = sigma_max * std::pow( double(i)/PML_L, PML_M );
    C01[i] = (1.0/Dt - 0.5*sigma0) / (1.0/Dt + 0.5*sigma0);
    C02[i] = 1.0 / (1.0/Dt + 0.5*sigma0);
  }

  for(int i = 0; i < PML_L; i++){
    double sigma1 = sigma_max * std::pow( (i + 0.5)/PML_L, PML_M );
    C11[i] = (1.0/Dt - 0.5*sigma1) / (1.0/Dt + 0.5*sigma1);
    C12[i] = 1.0 / (1.0/Dt + 0.5*sigma1);
  }

}



