#include "fdtd2d.h"

double ***allocate_memory3d(int l, int m, int n, double ini_v){
  double ***v = new double** [l];
  v[0] = new double *[l*m];
  v[0][0] = new double [l*m*n];
  for(int i = 0; i < l; i++){
    v[i] = v[0] + i*m;
    for(int j = 0; j < m; j++){
      v[i][j] = v[0][0] + i*m*n +j*n;
      for(int k = 0; k < n; k++){
        v[i][j][k] = ini_v;
      }
    }
  }
  return v;
}

double **allocate_memory2d(int m, int n, double ini_v){
  double **v = new double* [m];
  v[0] = new double [m*n];
  for(int i = 0; i < m; i++){
    v[i] = v[0] + i*n;
    for(int j = 0; j < n; j++){
      v[i][j] = ini_v;
    }
  }
  return v;
}

void free3d(double ***X, int l, int m){
  delete [] X[0][0];
  delete [] X[0];
  delete [] X;
}

void free2d(double **X, int m){
  delete [] X[0];
  delete [] X;
}
