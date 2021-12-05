#include "GA.h"
#include "agent.h"

void Agent::set_parameter(){
    // parameter_beta_1 = parameter_beta_1_min + parameter_beta_1_step + bin2dec(N_bit_parameter_beta_1, Gene);
    // parameter_beta_2 = parameter_beta_2_min + parameter_beta_2_step + bin2dec(N_bit_parameter_beta_2, Gene + N_bit_parameter_beta_1);
    // parameter_h_prime_1 = parameter_h_prime_1_min + parameter_h_prime_1_step + bin2dec(N_bit_parameter_h_prime_1, Gene + N_bit_parameter_beta_1 + N_bit_parameter_h_prime_1);
    // parameter_h_prime_2 = parameter_h_prime_2_min + parameter_h_prime_2_step + bin2dec(N_bit_parameter_h_prime_2, Gene + N_bit_parameter_beta_1 + N_bit_parameter_h_prime_1+N_bit_parameter_h_prime_1);

    parameter_beta_1 = parameter_beta_1_min + parameter_beta_1_step * bin2dec(0, N_bit_parameter_beta_1, Gene); /*始まりのビット、終わりのビット、遺伝子情報ビット*/
    parameter_beta_2 = parameter_beta_2_min + parameter_beta_2_step * bin2dec(N_bit_parameter_beta_1,N_bit_parameter_beta_1 + N_bit_parameter_beta_2, Gene);
    parameter_h_prime_1 = parameter_h_prime_1_min + parameter_h_prime_1_step * bin2dec(N_bit_parameter_beta_1 + N_bit_parameter_beta_2, N_bit_parameter_beta_1 + N_bit_parameter_beta_2 + N_bit_parameter_h_prime_1, Gene);
    parameter_h_prime_2 = parameter_h_prime_2_min + parameter_h_prime_2_step * bin2dec(N_bit_parameter_beta_1 + N_bit_parameter_beta_2 + N_bit_parameter_h_prime_1, N_bit_parameter_beta_1 + N_bit_parameter_beta_2 + N_bit_parameter_h_prime_1 + N_bit_parameter_h_prime_2, Gene);


}