#include <cmath>
#include <iostream>
#include "fdtd2d.h"


/*-----------------更新(1)-----------------*/
void update_Dr(double ***Dr, double **Hph, const int NEW, const int OLD){

#pragma omp paralell for
  for(int i = 0; i < Nr; i++){ //Dr(i+1/2,j)
    Dr[NEW][i][0] = Dr[OLD][i][0] + 4.0*Dt/r(i+0.5)/dth * Hph[i][0];

    for(int j = 1; j < Nth - PML_L; j++){/////θ=0
      Dr[NEW][i][j] = Dr[OLD][i][j] +
          Dt/r(i+0.5)/sin(j*dth)/dth *
          ( sin((j+0.5)*dth) * Hph[i][j] - sin((j-0.5)*dth) * Hph[i][j-1] );
    }
  }
}

void update_Dth(double ***Dth, double **Hph, const int NEW, const int OLD){
#pragma omp paralell for
  for(int i = 1; i < Nr; i++){ //Dth(i,j+1/2)
    for(int j = 0; j < Nth; j++){
      Dth[NEW][i][j] = Dth[OLD][i][j] - Dt/r(i)/dr *
          ( r(i+0.5) * Hph[i][j] - r(i-0.5) * Hph[i-1][j] );///
    }
  }
}

void update_Dph(double ***Dph, double **Hr, double **Hth, const int NEW, const int OLD){
#pragma omp paralell for
  for(int i = 1; i < Nr; i++){ //Dph(i,j)
    for(int j = 1; j < Nth - PML_L; j++){/////θ=0
      Dph[NEW][i][j] = Dph[OLD][i][j] + Dt/r(i)/dr *
          ( r(i+0.5) * Hth[i][j] - r(i-0.5) * Hth[i-1][j] )
          - Dt / (r(i)*dth) * ( Hr[i][j] - Hr[i][j-1] );
    }
  }
}
