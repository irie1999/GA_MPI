/*
 * suffix.cpp
 *
 *  Created on: 2021/06/23
 *      Author: ando
 */
#include <string>

#include "fdtd2d.h"

inline int round0(double x){
  return int( x+0.5 );
}

std::string suffix(double Lp, double z_dec, double sig_per){

  constexpr double m2km { 1e-3 };
  Lp *= m2km;
  z_dec *= m2km;
  sig_per *= m2km;

  std::string str_z_dec = std::to_string(int(z_dec)) + "."
      + std::to_string( int((z_dec - int(z_dec))*100) );
  std::string str_sig_per = std::to_string(int(sig_per)) + "."
      + std::to_string( int((sig_per - int(sig_per))*100) );

  std::string str_suffix = "_" + std::to_string(round0(Lp))
      + "_" + str_z_dec + "_" + str_sig_per;

  return str_suffix;
}

