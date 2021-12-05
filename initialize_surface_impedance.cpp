/*
 * initialize_surface_impedance.cpp
 *
 *  Created on: 2021/06/22
 *      Author: ando
 */
#include <iostream>
#include <cmath>

#include "fdtd2d.h"

/* とりあえず一様な導電率 */
double sig_s(double l){
  return 1.0e-3;
}

void initialize_surface_impedance(double *Rs, double *Ls){
  const double eps_r0 { Refractive_index(0.0)*Refractive_index(0.0) };

  for(int j = 0; j <= Nth; j++){
    double sig0 = sig_s(R0*j*dth);
    double r = std::sqrt( eps_r0*eps_r0 +
        sig0*sig0 / OMG / OMG / EPS0 / EPS0 );
    double th_arg = atan2( sig0, OMG * EPS0 * eps_r0 );

    Rs[j] = Z0 / std::sqrt(r) * std::cos(th_arg/2.0);
    Ls[j] = Z0 / OMG / std::sqrt(r) * std::sin(th_arg/2.0);
  }

}



