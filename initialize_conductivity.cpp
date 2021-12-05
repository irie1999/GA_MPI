/*
 * initialize_conductivity.cpp
 *
 *  Created on: 2021/06/02
 *      Author: ando
 */
#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <eigen3/Eigen/Dense>

//#include <iri2016.h>
#include "vector3d.h"

#include "fdtd2d.h"

extern std::string data_dir;

/*master*/
 double N(const double z_in_m, const double h_prime, const double beta){ //z[m] -> [km]
   return 1.43e13 * exp( -0.15 * h_prime * M2KM ) * exp( (beta-0.15) * (z_in_m - h_prime) * M2KM );  /*M2KM = 1.0e-3*/
 }
/*betaとh*/
// double N(const double z_in_m, double* h_prime, double* beta, int t){ //z[m] -> [km]
//   return 1.43e13 * exp( -0.15 * h_prime[t] * M2KM ) * exp( (beta[t]-0.15) * (z_in_m - h_prime[t]) * M2KM );  /*M2KM = 1.0e-3*/
// }

double nu(const double z_in_m){ //z[m] -> [km]
  return 4.303e11 * exp( -0.1622 * z_in_m / (1.0e3) );
}

/*master*/
 double omg_p(const double z, const double h_prime, const double beta){ //z[m]
  return sqrt( N(z, h_prime, beta)*CHARGE_e*CHARGE_e/(MASS_e*EPS0) );
 }
/*IRI*/
// double omg_p(int z, double *Ne){ //z[m]
//   return sqrt( Ne[z] * CHARGE_e * CHARGE_e / (MASS_e * EPS0) );
// }
/*betaとh*/
// double omg_p(const double z, double* h_prime, double* beta, int t){ //z[m]
//   return sqrt( N(z, h_prime, beta, t)*CHARGE_e*CHARGE_e/(MASS_e*EPS0) );
//}

/*master*/
void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
    const double h_prime,
    const double beta,
    const double Lp, const double z_dec, const double sig_per){
/*IRI*/
  // void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
  //    const double h_prime,
  //    const double beta,
  //    const double Lp, const double z_dec, const double sig_per, double *Ne){
/*betaとh*/
// void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
//      double* h_prime,
//      double* beta,
//      const double Lp, const double z_dec, const double sig_per,int t){
  std::complex <double> zj{ 0.0, 1.0 };

  /* 送受信点座標 */
  ANDO_LAB::vector3d <double> Tx =
      ANDO_LAB::geographic_coordinate(Tx_Latitude, Tx_Longitude);
  ANDO_LAB::vector3d <double> Rx =
      ANDO_LAB::geographic_coordinate(Rx_Latitude, Rx_Longitude);

  /* 磁場 */
  ANDO_LAB::vector3d <double> B0_geo = F0 * (
      - std::sin(Inc) * r_vector(Tx)
      - std::cos(Inc)*std::cos(Dec) * theta_vector(Tx)
      + std::cos(Inc)*std::sin(Dec) * phi_vector(Tx) );

  ANDO_LAB::vector3d <double> vec_n = (Tx * Rx).unit_vector(); /* φ方向 */
  ANDO_LAB::vector3d <double> vec_p = vec_n * Tx; /* θ方向 */

  /* 伝搬パス中央の方向ベクトル */
  ANDO_LAB::vector3d <double>
  Mid(1, Nth/2*dth, 0.0, ANDO_LAB::coordinate::Spherical );
//  Mid(1, M_PI/2.0, 0.0, ANDO_LAB::coordinate::Spherical );

  /* 磁場。とりあえず一様（球対称） */
  ANDO_LAB::vector3d <double> B0 =
  -F0*std::sin(Inc)*Mid + (B0_geo%vec_p)*theta_vector(Mid) + (B0_geo%vec_n)*phi_vector(Mid);

  /* 磁場の方向 */
  const double THE0 { B0.theta() };
  const double PHI0 { B0.phi() };
  std::cout << THE0 * 180 / M_PI << ", " << PHI0 * 180 / M_PI << std::endl;

  Eigen::Matrix3d R1, R2;
  R1<< cos(THE0), 0.0, sin(THE0),
      0.0,        1.0, 0.0,
      -sin(THE0), 0.0, cos(THE0);
  R2<<cos(PHI0), -sin(PHI0), 0.0,
      sin(PHI0),  cos(PHI0), 0.0,
      0.0,        0.0,       1.0;

  Eigen::Matrix3d p, q;
  p <<0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      -1.0,0.0, 0.0;
  q <<0.0, 0.0, -1.0,
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0;

  //constexpr double m2km { 1e-3 };

//  iri2016 iri;
//  iri.set_coord(35.0f, 142.0f); /* geographic coordinate */
//  iri.set_datetime(2015, 10, 1, 4, 0); /* UT */
//  iri.set_height(float(Lower_boundary_of_ionosphere*m2km), float(Rr*m2km), float(dr*m2km));

//  float *Ne = new float [Nr_iono];
//  iri.get_Ne(Ne);


  std::string filename = data_dir + "N" + suffix(Lp, z_dec, sig_per) + ".dat";
  std::ofstream ofs(filename.c_str());
  for(int i = 0; i < Nr_iono; i++){ //Er(i+1/2,j)
    double z = Lower_boundary_of_ionosphere + i*dr;

    double eps_r = Refractive_index(z) * Refractive_index(z);
//    ofs << z*m2km << " " << Ne[i] << std::endl;
//    const double Omg_p { sqrt(Ne[i])*CHARGE_e*CHARGE_e/(MASS_e*EPS0) };

    for(int j = 0; j <= Nth; j++){
      double path_length = j*dth * R0;
       double z_Ne = z + z_dec * std::exp( - (path_length - Lp)*(path_length - Lp)
           / 2.0 / sig_per / sig_per );
      

//      ofs << path_length*1e-3 << " " << (Lower_boundary_of_ionosphere + i*dr)*1e-3
//          << " " << N(z) << "\n";

      /*master*/
      const double Omg_p { omg_p(z_Ne, h_prime, beta) };
      /*IRI*/
      // double Omg_p;
      // if(i%2==0){
      //   Omg_p = omg_p(i/2, Ne);
      // }
      // else{
      //    Omg_p = omg_p((i-1)/2, Ne);
      // }
      /*betaとh*/
      //const double Omg_p { omg_p(z_Ne, h_prime, beta, t) };

      std::complex <double> omg_prime { OMG - zj*nu(z) };
      std::complex <double> alpha { omg_prime / (OMG_c*OMG_c - omg_prime*omg_prime) };
      std::complex <double> beta { zj * OMG_c / (OMG_c*OMG_c - omg_prime*omg_prime) };
      Eigen::Matrix3cd SIGz;
      SIGz <<
          alpha, beta,  0.0,
          -beta, alpha, 0.0,
          0.0,   0.0,   -1.0/omg_prime;
      SIGz = (zj * EPS0 * Omg_p * Omg_p) * SIGz;

      Eigen::Matrix3d SIGcar = R2 * R1 * SIGz.real() * R1.inverse() * R2.inverse();
      Eigen::Matrix3d SIG = p * SIGcar * q;

      /////(A,B,)C,Fの設定/////
      Eigen::Matrix3d A;
      A = EPS0*eps_r/Dt * Eigen::Matrix3d::Identity() + 0.5 * SIG;
      Eigen::Matrix3d B;
      B = EPS0*eps_r/Dt * Eigen::Matrix3d::Identity() - 0.5 * SIG;

      C[i][j] = A.inverse() * B;
      F[i][j] = (1.0/Dt) * A.inverse();
    }

//    ofs << "\n";
  }

  ofs.close();
}



