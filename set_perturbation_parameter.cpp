/*
 * void set_perturbation_parameter.cpp
 *
 *  Created on: 2021/06/23
 *      Author: ando
 */
#include <iostream>
#include <string>

#include "fdtd2d.h"

void set_perturbation_parameter(int argc, char **argv,
    double &Lp, double &z_dec, double &sig_per){

  /* check usage */
  if ( argc < 4 ){
    std::cout << "Error: not enough arguments.\n";
    std::cout << "Usage: main Lp z sigma\n";
    std::cout << " * Lp[km]: The center of the perturbation measured from the current source\n";
    std::cout << " * z_dec[km]: decreasing height of the electron density profile\n";
    std::cout << " * sigma[km]: the standard deviation of the perturbation\n";
    exit(0);
  }

  std::string str_Lp( argv[1] );
  std::string str_z_dec( argv[2] );
  std::string str_sig_per( argv[3] );

  constexpr double km2m { 1e3 };

  Lp = std::stod( str_Lp ) * km2m;
  z_dec = std::stod( str_z_dec ) * km2m;
  sig_per = std::stod( str_sig_per ) * km2m;


}
