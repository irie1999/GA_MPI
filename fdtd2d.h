#include <cmath>
#include <eigen3/Eigen/Dense>

/* Physical constants */
constexpr double C0 { 3.0e8 };
constexpr double MU0 { 4.0 * M_PI * 1e-7 };
constexpr double EPS0 { 1.0 / MU0 / C0 / C0 };
constexpr double Z0 { std::sqrt( MU0 / EPS0 ) };

constexpr double CHARGE_e { 1.602e-19 }; //[C]
constexpr double MASS_e { 9.1e-31 }; //[kg]

constexpr double R0 { 6370.0e3 }; /* Radius of Earth */
constexpr double M2KM { 1.0e-3 };

/*GA parameter*/
constexpr int Nr_ini { 100 };
constexpr int Nr_end { 900 };
constexpr int Nr_GA { Nr_end - Nr_ini };

/************************************************************
 * Setting parameters (from here)
 ************************************************************/
constexpr double Rr  { 120.0e3 }; /* Analysis Region in r-direction */
//constexpr double Rth { 2200.0e3 }; /* Analysis Region in θ-direction */
constexpr double Rth { 1000.0e3 }; /* Analysis Region in θ-direction */
//constexpr double dr { 0.25e3 };  /* Cell size in r */
constexpr double dr { 0.5e3 };
//constexpr double Rdth { 0.25e3 }; /* Cell size in θ */
constexpr double Rdth { 0.5e3 };


constexpr double Tmax { 0.00424 };

constexpr double FREQ { 22.2e3 }; /* [Hz] */
//constexpr double FREQ { 40.0e3 }; /* [Hz] */

/* 日本（えびのあたり？）の磁場 */
constexpr double Dec { -7.0 / 180 * M_PI };  // 偏角
constexpr double Inc { 49.0 / 180 * M_PI };  // 伏角
constexpr double F0 { 4.647e-5 };

/* 日本（おおたかどや山）の磁場 */
//constexpr double Dec { -7.53 / 180 * M_PI };  // 偏角 (>0East, <0West)
//constexpr double Inc { 51.25 / 180 * M_PI };  // 伏角
//constexpr double F0 { 4.7327e-5 };

/* えびの 緯度経度 */
constexpr double Tx_Latitude { 32.067650 };
constexpr double Tx_Longitude { 130.828978 };

/* おおたかどや山 緯度経度 */
//constexpr double Tx_Latitude { 37.372097972301226 };
//constexpr double Tx_Longitude { 140.84913351739013 };

/* 調布 緯度経度 */
constexpr double Rx_Latitude { 35.65966 };
constexpr double Rx_Longitude { 139.54328 };

/* Daytime condition */
//constexpr double z_prime { 73.0 }; //[km]
//constexpr double beta { 0.3 };

/* Nighttime condition */
//constexpr double z_prime { 85.0 }; //[km]
//constexpr double beta { 0.63 };

/* Current source Pulse waveform */
//constexpr double s  { 1.59099e-5 };
constexpr double s  { 1.0/2.0/M_PI/FREQ };
constexpr double t0 { 6.0 * s   };

/* PML */
constexpr int PML_L { 10 };
constexpr double PML_M { 3.2 };
constexpr double Gamma { 1.0e-6 };

/************************************************************
 * Setting parameters (up to here)
 ************************************************************/


constexpr double dth { Rdth/R0 }; /* Angle making a cell */
constexpr int Nr { int(Rr/dr) };  /* Cell number in r */
constexpr int Nth { int(Rth/Rdth) }; /* Cell number in theta */ /*Nth=2000, Rth=1000.0e3, Rdth=0.5e3*/

constexpr double Dt { 0.9 / C0 / sqrt( 1/(dr*dr) + 1/((R0*dth)*(R0*dth)) ) }; //0.999
//constexpr int Nt { int(Tmax/Dt)+1 };          //時間ループ
constexpr int Nt { 4000 };

constexpr double OMG { FREQ * 2. * M_PI }; //2*PI*f

/* Parameters of ionosphere */
//constexpr double Lower_boundary_of_ionosphere { 50.0e3 };
constexpr double Lower_boundary_of_ionosphere { 60.0e3 }; /*電子密度が60kmのため*/

constexpr int Nr_iono { int( (Rr - Lower_boundary_of_ionosphere) / dr ) + 1 };  /*121*/
constexpr int Nr_atmo { int( Lower_boundary_of_ionosphere / dr ) };

constexpr double OMG_c { CHARGE_e * F0 / MASS_e };

double I(double t);
double Jr(double t);

constexpr int m{ 10 }; /*m分間隔*/
//---------------tensor---------------//

//initialize
//double ***allocate_memory3d(int l, int m, int n, double ini_v);
//double **allocate_memory2d(int m, int n, double ini_v);

//forupdate2_E
Eigen::Matrix3d define_SIGzr(double z);

//update1_D
void update_Dr(double ***Dr, double **Hph, const int NEW, const int OLD);
void update_Dth(double ***Dth, double **Hph, const int NEW, const int OLD);
void update_Dph(double ***Dph, double **Hr, double **Hth, const int NEW, const int OLD);
void update_Dr_PML(double **Dr, double **Dr1, double **Dr2,
    double **Hph, double *C1, double *C2);
void update_Dph_PML(double **Dph, double **Dph_r, double **Dph_th,
    double **Hth, double **Hr,
    double *C1, double *C2);

//update2_E
void update_Er(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d **C, Eigen::Matrix3d **F);
void update_Eth(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d **C, Eigen::Matrix3d **F);
void update_Eph(double ***Er, double ***Eth, double ***Eph,
    double ***Dr, double ***Dth, double ***Dph,
    const int NEW, const int OLD,
    Eigen::Matrix3d **C, Eigen::Matrix3d **F);

//update3_H
void update_Hr(double **Hr, double **Eph);
void update_Hth(double **Hth, double **Eph, double *Rs, double *Ls);
void update_Hph(double **Hph, double **Er, double **Eth, double *Rs, double *Ls);
void update_Hr_PML(double **Hr, double **Hr1, double **Hr2,
    double **Eph, double *C1, double *C2);
void update_Hph_PML(double **Hph, double **Hph_r, double **Hph_th,
    double **Er, double **Eth, double *C1, double *C2,
    double *Rs, double *Ls, double **Bph, double *Bph_r, double *Bph_th, const int NEW);

/*master*/
void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
    const double z_prime,
    const double beta,
    const double Lp, const double z_dec, const double sig_per);

// /*IRI*/
// void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
//      const double z_prime,
//      const double beta,
//      const double Lp, const double z_dec, const double sig_per, double *Ne);

/*betaとh*/
// void initialize_conductivity(Eigen::Matrix3d** C, Eigen::Matrix3d** F,
//      double* z_prime,
//      double* beta,
//      const double Lp, const double z_dec, const double sig_per,int t);

void initialize_pml(double *C01, double *C02, double *C11, double *C12);

void initialize_surface_impedance(double *Rs, double *Ls);

void set_perturbation_parameter(int argc, char **argv,
    double &Lp, double &z_dec, double &sig_per);

std::string suffix(double Lp, double z_dec, double sig_per);

//input
/*IRI*/
//void input(double *Ne, int t, int s);
/*beta_h*/
//void input(double *beta, double *z_prime);

//output
void output(double ***Er, int NEW, int n);

//free
//void free3d(double ***X, int l, int m);
//void free2d(double **X, int m);

inline double r(double i){
  return R0 + i*dr;
}

inline double Refractive_index(const double z){
  /* refractive index of standard air */
  return 1.000325 - 0.039e-6 * z;
}
