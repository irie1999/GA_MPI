#include <iostream>
#include <fstream>
#include <cmath>

#include "fdtd2d.h"
void input(double **S, int time){
	/*ファイル読み込み*/
	std::ifstream ifs;  // ファイル読み取り用ストリーム  
	ifs.open("../data/Si_tm_" + std::to_string(time) + ".dat");	// ファイルオープン

	if(ifs.fail()){	// ファイルオープンに失敗したらそこで終了
		std::cerr << "ファイルを開けません\n";
		exit(1);
	}

	char buf[801];	// データ一時保管用配列

	int linenum = 0; // データの行数を数える
	char *p;
	double *ggg = new double [900]; /*テスト用*/

	while(ifs.getline(buf,sizeof(buf))){	// ファイルから1行ずつ読み込む
		linenum++;	// 行数をカウントしている
	}

	std::cerr << "読み込んだ行数 = " << linenum << "\n";

	ifs.clear(); // ファイル末尾に到達というフラグをクリア
	ifs.seekg(0, std::ios::beg);	// ファイル先頭に戻る

	for(int i=0 ; i<linenum ; i++){
		ifs.getline(buf,sizeof(buf));	// 一行読み込んで…
        p = strtok(buf," ");
        ggg[i] = atof(p);
        p = strtok(NULL, "\n");
        S[time][i] = atof(p);
		
	}
}
