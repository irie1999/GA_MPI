#include <random>
#include <fstream>

#include "GA.h"
#include "agent.h"

#include "memory_allocate.h"


double fitting(double parameter_beta_1, double parameter_beta_2, 
            double parameter_h_prime_1, double parameter_h_prime_2, double **S){
    double v = 0.0; /*score*/
    double **s = allocate_memory2d(3, 801, 0.0);  /*探索する個体の電界強度の変化率*/
    double **Ei_tm = allocate_memory2d(3, 801, 0.0); /*個体のβとh'を与えた得られた電界強度"*/
    double beta[3], h_prime[3];
    double time[3];
    //double compare_1, compare_2, compare_3;  /*評価関数の3つの部分を比較する変数*/
    time[0] = 6.1667;
    time[1] = 6.313;
    time[2] = 6.5; 
    beta[0] = 0.49366;
    h_prime[0] = 77.69128e3;

    for(int t = 0; t <= 2; t++){
        beta[t] = parameter_beta_2 * pow((time[t] - time[0]), 2) + parameter_beta_1 * (time[t] - time[0]) + beta[0];
        h_prime[t] = parameter_h_prime_2 * 1e3 * pow((time[t] - time[0]), 2) + parameter_h_prime_1 * 1e3 * (time[t] - time[0]) + h_prime[0];
        std::cout << "t = " << t << std::endl;
        cal_fdtd(beta[t], h_prime[t], t, Ei_tm); /*betaとh'を代入して電界を返す*/
    }
    

    for(int t_m = 1; t_m <= M; t_m++){
        for(int i = 0; i < GA_Nr; i++){
            s[t_m][i] = (Ei_tm[t_m][i] - Ei_tm[t_m - 1][i]) / (time[t_m] - time[t_m -1]);

            v += 1/ (GA_Nr * (M - 1)) * std::pow(std::abs( S[t_m][i] - s[t_m][i] ), 2);
        }
        
            v += p_beta * std::pow(std::abs( beta[M] - beta[0]), 2) 
                 + p_h_prime * std::pow(std::abs( h_prime[M] - beta[0]), 2);
    }
    std::ofstream ofs("../data/" "compare" + std::to_string(Number_of_Generation) + "," + std::to_string(Number_of_Individual) + ".dat");
    ofs << "compare_1= " << v - p_beta * std::pow(std::abs( beta[M] - beta[0]), 2)  + p_h_prime * std::pow(std::abs( h_prime[M] - beta[0]), 2)
        << std::endl << "compare_2= " << std::pow(std::abs( beta[M] - beta[0]), 2) << std::endl 
        << "compare_3= " << std::pow(std::abs( h_prime[M] - beta[0]), 2) << std::endl;
    ofs.close();

    deallocate_memory2d(s);
    deallocate_memory2d(Ei_tm);
    delete[] beta;
    delete[] h_prime;
    delete[] time;

    return v;
}

void cal_ind(Agent *p, double **S, int i, double **parameter, double *score){
    std::cout << "個体= " << i << std::endl; 
    //p[i].set_parameter();  /* 2進数から10進数に変換*/
    //p[i].score  /*FDTDの計算,返値がスコア*/
    score[i] = fitting(parameter[0][i], parameter[1][i], parameter[2][i], parameter[3][i], S); 
                    
}

void create_ind(Agent *agent){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //std::mt19937 rnd(1); 
    for(int i = 0; i < Number_of_Individual; i++){
        for(int n = 0; n < N_bit_total; n++){
            agent[i].Gene[n] = rnd() % 2;
        }
    }
}




void compose_roulette(const int N, Agent *agent, double *roulette, double *score_average, int n_generation){/*ルーレット作成*/
    double sum = 0.0; 
    for(int i = 0; i < Number_of_Individual; i++){
        sum += agent[i].score;
    }
    score_average[n_generation] = sum / Number_of_Individual;
    roulette[0] = agent[0].score / sum;
    for(int i = 1; i < Number_of_Individual ; i++){
        roulette[i] = roulette[i-1] + agent[i].score / sum;
    }
}

void crossover(int head, Agent *p, Agent *c, int *s){ /*交叉*/
    //std::mt19937 rnd(1); 
    std::random_device rnd;
    std::mt19937 mt(rnd());
  
    for(int i = 0; i < N_bit_total; i++){
        if(rnd() / i32 < 0.5){  /*入れ替えなし*/  
            c[head].Gene[i] = p[s[0]].Gene[i];  
            c[head+1].Gene[i] = p[s[1]].Gene[i];
        }
        else{                   /*入れ替えあり*/
            c[head].Gene[i] = p[s[1]].Gene[i];
            c[head+1].Gene[i] = p[s[0]].Gene[i];
        }
    }
}

void selection_crossover(double *roulette, Agent *p, Agent *c){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //std::mt19937 rnd(1);
    for(int i = 2; i < Number_of_Individual; i+=2){
            int sict[2];
            for(int j = 0; j <2 ; j++){
                double rnd_num = rnd() / i32;
                int k = 0;
                while( roulette[k] < rnd_num){  /*親を2体選ぶルーレット*/
                    k++;
                }
                sict[j] = k;
            }
            crossover(i, p, c, sict); /*交叉*/
    }
    /*エリート戦略*/
    for(int i = 0; i < 2; i++){ /*スコア上位2体を無条件に選択*/
        for(int n = 0; n < N_bit_total; n++){
            c[i].Gene[n] = p[i].Gene[n];
        }
    }
}

void mutate_ind(Agent *c){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //std::mt19937 rnd(1);
    for(int i = 0; i < Number_of_Individual; i++){
        for (int j = 0; j < N_bit_total; j++){
            if(rnd() / i32 < MUTATION ){
                c[i].Gene[j] = !c[i].Gene[j];  /*0と1を反転*/
            }
        }
    }
}

void final_cal_ind(Agent *p, double *max_parameter, double *MAX, double *score_average, double **S){
    double sum = 0.0;
    for(int i = 0; i < Number_of_Individual; i++){
        p[i].set_parameter();  /* 2進数から10進数に変換*/
        p[i].score/*FDTDの計算,返値がスコア*/
            = fitting( p[i].parameter_beta_1, p[i].parameter_beta_2,
                       p[i].parameter_h_prime_1, p[i].parameter_h_prime_2, S); 
        sum += p[i].score;
        sort_ind(p);
    }
    MAX[Number_of_Generation - 1] = p[0].score;
    score_average[Number_of_Generation - 1] = sum / (Number_of_Individual - 1);
    max_parameter[0] = p[0].parameter_beta_1;
    max_parameter[1] = p[0].parameter_beta_2;
    max_parameter[2] = p[0].parameter_h_prime_1;
    max_parameter[3] = p[0].parameter_h_prime_2;
}

void sort_ind(Agent *p){
    Agent tmp;
    for(int i = 0; i < Number_of_Individual - 1; i++){
        for(int j = i + 1 ; j < Number_of_Individual; j++){
            if( p[i].score < p[j].score ){
                tmp = p[i];
                p[i] = p[j];
                p[j] = tmp;

            } 
        }
    }
}


/*2進数→10進数変換*/
int bin2dec(const int N_bit_initial, const int N_bit_end, bool *binary){
    int v { 0 };
    int base { 1 };

    for(int i = N_bit_end - 1; i >= N_bit_initial; i--){
        v += int(binary[i]) * base;
        base *= 2;
    }
    return v;
}

// double **allocate_memory2d(int m, int n, double ini_v){
//   double **v = new double* [m];
//   v[0] = new double [m*n];
//   for(int i = 0; i < m; i++){
//     v[i] = v[0] + i*n;
//     for(int j = 0; j < n; j++){
//       v[i][j] = ini_v;
//     }
//   }
//   return v;
// }

