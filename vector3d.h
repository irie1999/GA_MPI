/*
 * vector3d.h
 *
 *  Created on: 2021/02/12
 *      Author: ando
 *
 * 3次元ベクトルのテンプレート
 * ※現在はデカルト座標系と球座標系のみ実装
 *
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <cmath>

namespace ANDO_LAB{

constexpr double DEG2RAD { M_PI / 180.0 };
constexpr double RAD2DEG { 180.0 / M_PI };
enum class coordinate { Cartesian, Cylindrical, Spherical };

template <class T>
class vector3d{
private:
  T pX, pY, pZ;
  T pR, pTheta, pPhi;

  /* 座標変換 */
  void convert2spherical(void);
  void convert2cartesian(void);

public:
  /* コンストラクタ */
  vector3d(T v1 = {0.0}, T v2 = {0.0}, T v3 = {0.0}, coordinate = { coordinate::Cartesian });

  /* アクセサ */
  void set(T v1, T v2, T v3, coordinate = { coordinate::Cartesian });
  T x(void){ return pX; };
  T y(void){ return pY; };
  T z(void){ return pZ; };
  T r(void){ return pR; };
  T theta(void){ return pTheta; };
  T phi(void){ return pPhi; };

  T abs(void){ return pR; }; /* 大きさ */
  vector3d <T> unit_vector(void); /* 単位ベクトル */

  template <class T2>
  friend vector3d <T2> geographic_coordinate( /* 緯度経度を単位ベクトルに変換 */
      T2 Latitude_in_deg, T2 Longitude_in_deg);

  template <class T2>
  friend vector3d <T2> r_vector( /* 指定した位置のr方向単位ベクトル */
      vector3d <T2> r);
  template <class T2>
  friend vector3d <T2> theta_vector( /* 指定した位置のθ方向単位ベクトル */
      vector3d <T2> r);
  template <class T2>
  friend vector3d <T2> phi_vector( /* 指定した位置のφ方向単位ベクトル */
      vector3d <T2> r);

  /* 演算子 */
  vector3d <T> operator + (vector3d <T>); /* ベクトル和 */
  vector3d <T> operator - (vector3d <T>); /* ベクトル差 */
  vector3d <T> operator *(T);  /* スカラ倍 */
  vector3d <T> operator *(vector3d <T>);  /* 外積 */
  T operator %(vector3d <T>);  /* 内積 */

  template <class T2>
  friend vector3d <T2> operator * (T2, vector3d <T2>);  /* スカラ倍 */
};

/* コンストラクタ */
template <class T>
vector3d <T>::vector3d(T v1, T v2, T v3, coordinate coord){
  set(v1, v2, v3, coord);
}

/* アクセサ */
template <class T>
void vector3d <T>::set( /* ベクトルを設定する */
    T v1, T v2, T v3, /* 3成分(x,y,z), (r,θ,φ) */
    coordinate coord  /* 座標系を指定（デフォルトはデカルト座標系） */
    ){
  if ( coord == coordinate::Cartesian ){
    pX = v1;
    pY = v2;
    pZ = v3;
    convert2spherical();
  } else if ( coord == coordinate::Spherical ){
    pR = v1;
    pTheta = v2;
    pPhi = v3;
    convert2cartesian();
  }
}

/* 座標変換 */
template <class T>
void vector3d <T>::convert2spherical(void){
  pR = std::sqrt(pX*pX + pY*pY + pZ*pZ);
  pPhi = std::atan2( pY, pX );
  pTheta = std::atan2( std::sqrt(pX*pX + pY*pY), pZ );
}

template <class T>
void vector3d <T>::convert2cartesian(void){
  pX = pR * std::sin(pTheta) * std::cos(pPhi);
  pY = pR * std::sin(pTheta) * std::sin(pPhi);
  pZ = pR * std::cos(pTheta);
}

/* 単位ベクトル */
template <class T>
vector3d <T> vector3d <T>::unit_vector(void){
  return vector3d <T> { 1.0, pTheta, pPhi, coordinate::Spherical };
}

/* 緯度経度を単位ベクトルに変換 */
template <class T>
vector3d <T> geographic_coordinate(T lat, T lon){
  T th = M_PI/2.0 - lat*DEG2RAD;
  T ph = lon * DEG2RAD;
  return vector3d <T> { 1.0, th, ph, coordinate::Spherical };
}

/* 点rにおける、r方向、θ方向、φ方向単位ベクトル */
template <class T>
vector3d <T> r_vector(vector3d <T> r){
  return r.unit_vector();
}

template <class T>
vector3d <T> theta_vector(vector3d <T> r){
  return vector3d <T> { 1.0, r.pTheta + M_PI/2.0, r.pPhi, coordinate::Spherical };
}

template <class T>
vector3d <T> phi_vector(vector3d <T> r){
  return vector3d <T> { 1.0, M_PI/2.0, r.pPhi + M_PI/2.0, coordinate::Spherical };
}

/* 演算子 */
template <class T>
vector3d <T> vector3d<T>::operator +(vector3d <T> op){
  return vector3d <T> (pX + op.pX, pY+op.pY, pZ+op.pZ);
}

template <class T>
vector3d <T> vector3d<T>::operator -(vector3d <T> op){
  return vector3d <T> (pX - op.pX, pY-op.pY, pZ-op.pZ);
}

template <class T>
vector3d <T> vector3d <T>::operator *(T coef){
  return vector3d <T> (coef*pX, coef*pY, coef*pZ);
}

template <class T>
vector3d <T> operator *(T coef, vector3d <T> op){
  return vector3d <T> (coef*op.pX, coef*op.pY, coef*op.pZ);
}

template <class T>
T vector3d <T>::operator % (vector3d <T> op){
  return pX*op.pX + pY*op.pY + pZ*op.pZ;
}

template <class T>
vector3d <T> vector3d <T>::operator *(vector3d <T> op){
  return vector3d <T> (
      pY*op.pZ - pZ*op.pY,
      pZ*op.pX - pX*op.pZ,
      pX*op.pY - pY*op.pX );
}


}
#endif /* VECTOR3D_H_ */
