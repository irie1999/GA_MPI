/*
 * memory_allocate.h
 *
 *  Created on: 2016/09/02
 *      Author: ando
 *
 *tex \section{多次元配列の確保テンプレート}
 *tex 本節の関数は全て、\textsf{memory\_allocate.h} をインクルードする。
 *tex $2 \leqq n \leqq 4$である。
 *
 */

#ifndef MEMORY_ALLOCATE_H_
#define MEMORY_ALLOCATE_H_

/*
 *tex \subsection{$n$次元配列の確保 \textsl{Type} 
 *tex   \textsf{$\ast\ast\ast$allocate\_memory}$n$\textsf{d(
 *tex     int x1, int x2, $\cdots$, int x}$n$\textsf{, \textsl{Type} v)}}
 *tex \begin{description}
 *tex   \item[説明]$\textsf{x1} \times \textsf{x2} \times \cdots \times \textsf{x}n$の
 *tex     \textsf{\textsl{Type}}型の$n$次元配列を確保し、ポインタを返す。
 *tex     その際、各要素の初期値は \textsf{v} となる。
 *tex \end{description}
 *tex 
 */
template <class Var>
Var ***allocate_memory3d
(int nx, int ny, int nz, Var v0){
  Var ***v = new Var ** [nx];
  v[0] = new Var *[nx*ny];
  v[0][0] = new Var [nx*ny*nz];

  for(int i = 0; i < nx; i++){
    v[i] = v[0] + i*ny;
    for(int j = 0; j < ny; j++){
      v[i][j] = v[0][0] + i*ny*nz + j*nz;
      for(int k = 0; k < nz; k++){
        v[i][j][k] = v0;
      }
    }
  }

  return v;
}

/*
 *tex 
 *tex \subsection{$n$次元配列の開放 \textsl{Type} \textsf{deallocate\_memory}$n$%
 *tex   \textsf{d(Type }\textrm{pointer})}
 *tex \begin{description}
 *tex   \item[説明]\textsf{\textsl{Type}}型の$n$次元配列を開放する。
 *tex \end{description}
 *tex 
 */
template <class Var>
void deallocate_memory3d(Var ***v){
  delete [] v[0][0];
  delete [] v[0];
  delete [] v;
}


template <class Var>
Var **allocate_memory2d
(int nx, int ny, Var v0){
  Var **v = new Var * [nx];
  v[0] = new Var [nx*ny];

  for(int i = 0; i < nx; i++){
    v[i] = v[0] + i*ny;
    for(int j = 0; j < ny; j++){
      v[i][j] = v0;
    }
  }

  return v;
}

template <class Var>
void deallocate_memory2d(Var **v){
  delete [] v[0];
  delete [] v;
}


template <class Var>
Var ****allocate_memory4d
(int nw, int nx, int ny, int nz, Var v0){
  Var ****v = new Var *** [nw];
  v[0] = new Var ** [nw*nx];
  v[0][0] = new Var * [nw*nx*ny];
  v[0][0][0] = new Var [nw*nx*ny*nz];

  for(int l = 0; l < nw; l++){
    v[l] = v[0] + l*nx;
    for(int i = 0; i < nx; i++){
      v[l][i] = v[0][0] + l*nx*ny + i*ny;
      for(int j = 0; j < ny; j++){
        v[l][i][j] = v[0][0][0] + l*nx*ny*nz + i*ny*nz + j*nz;
        for(int k = 0; k < nz; k++){
          v[l][i][j][k] = v0;
        }
      }
    }
  }

  return v;
}


template <class Var>
void deallocate_memory4d(Var ****v){
  delete [] v[0][0][0];
  delete [] v[0][0];
  delete [] v[0];
  delete [] v;
}



#endif /* MEMORY_ALLOCATE_H_ */
