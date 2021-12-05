#include <iostream>
#include <fstream>
#include "fdtd2d.h"


void output(double ***Er, int NEW, int n){
  std::ofstream ofs( ("data/er_" + std::to_string(n) + ".dat").c_str() );//////
  for(int i = 0; i < Nr; i+=5){
    for(int j = 0; j < Nth; j+=2){
      ofs << i << " " << j << " " << Er[NEW][i][j] << "\n";
    }
    ofs << "\n";
  }
  ofs.close();
//  std::cout << " n = " << n << "  出力完了... " << "\n";
}
