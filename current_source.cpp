#include "fdtd2d.h"


double I(double t){
  return (t-t0) / s * exp( - (t-t0)*(t-t0) / (2.0*s*s) );
}
double Jr(double t){   ///Jr(R0+(i+1/2)*Dr,j*Dth) = Jr(i+1/2,j)
  double R = (R0 + dr/2.0) * dth/2.0; /* radius of circular area at current density */
  const double Mag = 0.5e3/dr;
  return Mag * I(t) / M_PI / R / R;
}
