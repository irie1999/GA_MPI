#include <iostream>
#include "fdtd2d.h"


/*-----------------更新(3)-----------------*/
void update_Hr(double **Hr, double **Eph){
#pragma omp paralell for
  for(int i = 1; i < Nr; i++){ //Hr(i,j+1/2)
    for(int j = 0; j < Nth - PML_L; j++){
      Hr[i][j] = Hr[i][j] -
          Dt/MU0/r(i)/sin((j+0.5)*dth)/dth *
          ( sin((j+1)*dth) * Eph[i][j+1] - sin(j*dth) * Eph[i][j] );
    }
  }
}

void update_Hth(double **Hth, double **Eph, double *Rs, double *Ls){

#pragma omp paralell for
  for(int j = 1; j < Nth; j++){
    double c1 {
      (MU0*r(0.5)*dr - Dt*r(0.0) * (Rs[j]/2.0 - Ls[j]/Dt ) ) /
          (MU0*r(0.5)*dr + Dt*r(0.0) * (Rs[j]/2.0 + Ls[j]/Dt ) ) };

    double c2 {
      Dt * r(1.0) /
          (MU0*r(0.5)*dr + Dt*r(0.0) * (Rs[j]/2.0 + Ls[j]/Dt ) )
    };

    Hth[0][j] = c1 * Hth[0][j] + c2 * Eph[1][j];
  }

  for(int i = 1; i < Nr; i++){ //Hth(i+1/2,j)
    for(int j = 1; j < Nth; j++){/////θ=0
      Hth[i][j] = Hth[i][j] + Dt/MU0/r(i+0.5)/dr * ( r(i+1) * Eph[i+1][j] - r(i) * Eph[i][j]);
    }
  }
}

void update_Hph(double **Hph, double **Er, double **Eth, double *Rs, double *Ls){

#pragma omp paralell for
  for(int j = 0; j < Nth - PML_L; j++){
    double c1 {
      ( MU0 * r(0.5) * dr - Dt * r(0.0) * ((Rs[j]+Rs[j+1])/4.0 - (Ls[j]+Ls[j+1])/2.0/Dt) ) /
          ( MU0 * r(0.5) * dr + Dt * r(0.0) * ((Rs[j]+Rs[j+1])/4.0 + (Ls[j]+Ls[j+1])/2.0/Dt) )
    };

    double c2{
      Dt / ( MU0 * r(0.5) + Dt/dr * r(0.0) * ((Rs[j]+Rs[j+1])/4.0 + (Ls[j]+Ls[j+1])/2.0/Dt) )
    };

    Hph[0][j] = c1 * Hph[0][j] +
        c2 * ( -r(1.0)/dr * Eth[1][j] + (Er[0][j+1] - Er[0][j]) / dth );


  }

#pragma omp paralell for
  for(int i = 1; i < Nr; i++){ //Hph(i+1/2,j+1/2)
    for(int j = 0; j < Nth - PML_L; j++){
      Hph[i][j] = Hph[i][j] -
          Dt/MU0/r(i+0.5)/dr * ( r(i+1) * Eth[i+1][j] - r(i) * Eth[i][j]) +
          Dt/MU0/r(i+0.5)/dth * ( Er[i][j+1] - Er[i][j]);
    }
  }
}
