#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <mpi.h>

#include "GA.h"
#include "agent.h"
#include "memory_allocate.h"

int main(int argc, char **argv){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //std::mt19937 rnd(1); 

    MPI::Init(argc, argv);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int size = MPI::COMM_WORLD.Get_size();

    std::cout << "rank= " << rank << ", size= " << size << std::endl;
    
    double **S = allocate_memory2d(3, 801, 0.0);  /*観測した電界強度の変化率*/
    double max_parameter[4];  /*最終世代のスコアの最大値のパラメーターを格納*/ 
    double MAX[Number_of_Generation + 1];   /*最大値を格納*/
    double score_average[Number_of_Generation + 1]; /*平均値を格納*/
    double roulette[Number_of_Individual];  /*ルーレット*/
    Agent agent[2][Number_of_Individual];   /*個体*/
 
    create_ind(agent[0]); /*初期ランダム遺伝子の作成*/

    input(S,1);  /*t_1の時の観測した電界強度*/
    input(S,2);  /*t_2の時の観測した電界強度*/

    std::cout << "世代=" << Number_of_Generation << std::endl
              << "個体=" << Number_of_Individual << std::endl;
    
    for(int n_generation = 0; n_generation < Number_of_Generation - 1; n_generation++){
        std::cout << "世代= " << n_generation << std::endl;
        const int PARENT { n_generation % 2 };
        const int CHILD { (n_generation + 1) % 2};
        /*スコアを計算*/
        cal_ind(agent[PARENT], S);
        
        /*スコア順にソート*/
        sort_ind(agent[PARENT]);

        /*各世代の最大値を格納*/
        MAX[n_generation] = agent[PARENT][0].score;

        /*ルーレットと平均値作成*/
        compose_roulette(Number_of_Individual, agent[PARENT], roulette, score_average, n_generation);    
        
        /*選択と交叉*/
        selection_crossover(roulette, agent[PARENT], agent[CHILD]);
        
        /*突然変異*/
        mutate_ind(agent[CHILD]);
        
    }
    
    /*最終世代の最もスコアが高いものを判断*/
    const int PARENT { (Number_of_Generation - 1) % 2 };
   
    final_cal_ind(agent[PARENT], max_parameter, MAX, score_average, S);

    std::ofstream ofs("../data/" "score_graph");
    for(int n_generation = 0; n_generation < Number_of_Generation; n_generation++){
        ofs << n_generation << " " << MAX[n_generation] << " " << score_average[n_generation] << std::endl;
    }

    std::cout << "beta_1= " << max_parameter[0] << " beta_2= " << max_parameter[1]  << " h_prime_1= " << max_parameter[2]  
              << " h_prime_2= " << max_parameter[3] << " max= " << MAX[Number_of_Generation - 1] << std::endl;

    double beta[3], h_prime[3];
    double time[3];
    time[0] = 6.1667;
    time[1] = 6.313;
    time[2] = 6.5; 
    beta[0] = 0.49366;
    h_prime[0] = 77.69128;
    for(int t = 1; t < 3; t++){
    std::cout << "beta_" + std::to_string(t) << "= " << max_parameter[1] * pow((time[t] - time[0]),2) + max_parameter[0] * (time[t] - time[0]) + beta[0] << std::endl
              << "h_prime" + std::to_string(t)  << "= " << max_parameter[3] * pow((time[t] - time[0]),2) + max_parameter[2] * (time[t] - time[0]) + h_prime[0] << std::endl;
    }

    deallocate_memory2d(S);

    return 0;

}
