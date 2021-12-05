#include <iostream>

#include <eigen3/Eigen/Dense>
#include "fdtd2d.h"


/*-----------------更新(2)-----------------*/
void update_Er(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d** C, Eigen::Matrix3d** F){

  for(int i = 0; i < Nr_atmo; i++){
    for(int j = 0; j < Nth; j++){
      Er[NEW][i][j] = Dr[NEW][i][j] / EPS0;
    }
  }

#pragma omp parallel for
  for(int i = Nr_atmo; i < Nr; i++){ //Er(i+1/2,j)
    const int i_ofs = i - Nr_atmo;
    
    Er[NEW][i][0] =  0.5 * (C[i_ofs][0](0,0) + C[i_ofs+1][0](0,0)) * Er[OLD][i][0] +
        0.5 * (F[i_ofs][0](0,0) + F[i_ofs+1][0](0,0)) * (Dr[NEW][i][0]-Dr[OLD][i][0]); //θ=0

    for(int j = 1; j < Nth; j++){/////θ=0
      Er[NEW][i][j] =
          0.5* (C[i_ofs][j](0,0) + C[i_ofs+1][j](0,0))*Er[OLD][i][j] +
          0.5* (C[i_ofs][j](0,1) + C[i_ofs+1][j](0,1))*0.25*(Eth[OLD][i][j] + Eth[OLD][i][j-1] + Eth[OLD][i+1][j] + Eth[OLD][i+1][j-1]) +
          0.5* (C[i_ofs][j](0,2) + C[i_ofs+1][j](0,2))*0.5*(Eph[OLD][i][j] + Eph[OLD][i+1][j]) +
          0.5* (F[i_ofs][j](0,0) + F[i_ofs+1][j](0,0))*(Dr[NEW][i][j]  - Dr[OLD][i][j]) +
          0.5* (F[i_ofs][j](0,1) + F[i_ofs+1][j](0,1))*0.25*(Dth[NEW][i][j] + Dth[NEW][i][j-1] + Dth[NEW][i+1][j] + Dth[NEW][i+1][j-1] -
              Dth[OLD][i][j] - Dth[OLD][i][j-1] - Dth[OLD][i+1][j] - Dth[OLD][i+1][j-1]) +
              0.5* (F[i_ofs][j](0,2) + F[i_ofs+1][j](0,2))*0.5*(Dph[NEW][i][j] + Dph[NEW][i+1][j] - Dph[OLD][i][j] - Dph[OLD][i+1][j]);
    }

//    Er[NEW][i][0] =  C[i_ofs][0](0,0) * Er[OLD][i][0] +
//        F[i_ofs][0](0,0) * (Dr[NEW][i][0]-Dr[OLD][i][0]); //θ=0
//
//    for(int j = 1; j < Nth; j++){/////θ=0
//      Er[NEW][i][j] =
//          C[i_ofs][j](0,0)*Er[OLD][i][j] +
//          C[i_ofs][j](0,1)*0.25*(Eth[OLD][i][j] + Eth[OLD][i][j-1] + Eth[OLD][i+1][j] + Eth[OLD][i+1][j-1]) +
//          C[i_ofs][j](0,2)*0.5*(Eph[OLD][i][j] + Eph[OLD][i+1][j]) +
//          F[i_ofs][j](0,0)*(Dr[NEW][i][j]  - Dr[OLD][i][j]) +
//          F[i_ofs][j](0,1)*0.25*(Dth[NEW][i][j] + Dth[NEW][i][j-1] + Dth[NEW][i+1][j] + Dth[NEW][i+1][j-1] -
//              Dth[OLD][i][j] - Dth[OLD][i][j-1] - Dth[OLD][i+1][j] - Dth[OLD][i+1][j-1]) +
//          F[i_ofs][j](0,2)*0.5*(Dph[NEW][i][j] + Dph[NEW][i+1][j] - Dph[OLD][i][j] - Dph[OLD][i+1][j]);
//    }
  }

}


void update_Eth(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d** C, Eigen::Matrix3d** F){

#pragma omp parallel for
  for(int i = 1; i < Nr_atmo; i++){
    for(int j = 0; j < Nth; j++){
      Eth[NEW][i][j] = Dth[NEW][i][j] / EPS0;
    }
  }

#pragma omp parallel for
  for(int i = Nr_atmo; i < Nr; i++){ //Er(i,j+1/2)
    const int i_ofs = i - Nr_atmo;

    for(int j = 0; j < Nth; j++){
      Eth[NEW][i][j] =
          C[i_ofs][j](1,0)*0.25*(Er[OLD][i][j] + Er[OLD][i-1][j] + Er[OLD][i][j+1] + Er[OLD][i-1][j+1]) +
          C[i_ofs][j](1,1)*Eth[OLD][i][j] +
          C[i_ofs][j](1,2)*0.5*(Eph[OLD][i][j] + Eph[OLD][i][j+1]) +
          F[i_ofs][j](1,0)*0.25*(Dr[NEW][i][j] + Dr[NEW][i-1][j] + Dr[NEW][i][j+1] + Dr[NEW][i-1][j+1] -
              Dr[OLD][i][j] - Dr[OLD][i-1][j] - Dr[OLD][i][j+1] - Dr[OLD][i-1][j+1]) +
          F[i_ofs][j](1,1)*(Dth[NEW][i][j]-Dth[OLD][i][j]) +
          F[i_ofs][j](1,2)*0.5*(Dph[NEW][i][j] + Dph[NEW][i][j+1] -
              Dph[OLD][i][j] - Dph[OLD][i][j+1]);
    }
  }
}


void update_Eph(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d** C, Eigen::Matrix3d** F){

#pragma omp parallel for
  for(int i = 1; i < Nr_atmo; i++){
    for(int j = 1; j < Nth; j++){
      Eph[NEW][i][j] = Dph[NEW][i][j] / EPS0;
    }
  }

#pragma omp parallel for
  for(int i = Nr_atmo; i < Nr; i++){ //Eph(i,j)
    const int i_ofs = i - Nr_atmo;

    for(int j = 1; j < Nth; j++){/////θ=0
      Eph[NEW][i][j] =
          C[i_ofs][j](2,0)*0.5*(Er[OLD][i][j] + Er[OLD][i-1][j]) +
          C[i_ofs][j](2,1)*0.5*(Eth[OLD][i][j] + Eth[OLD][i][j-1]) +
          C[i_ofs][j](2,2)*Eph[OLD][i][j] +
          F[i_ofs][j](2,0)*0.5*(Dr[NEW][i][j] + Dr[NEW][i-1][j] - Dr[OLD][i][j] - Dr[OLD][i-1][j]) +
          F[i_ofs][j](2,1)*0.5*(Dth[NEW][i][j] + Dth[NEW][i][j-1] - Dth[OLD][i][j] - Dth[OLD][i][j-1]) +
          F[i_ofs][j](2,2)*(Dph[NEW][i][j]-Dph[OLD][i][j]);
    }

  }
}
