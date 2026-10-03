#include <Eigen/Core>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
 const int n=100001;Eigen::ArrayXf x(n);
 std::ifstream input(argv[1],std::ios::binary);input.read(reinterpret_cast<char*>(x.data()),n*sizeof(float));
 Eigen::ArrayXf t=x.tanh(),s=x.logistic();
 std::ofstream tanh(argv[2],std::ios::binary),sigmoid(argv[3],std::ios::binary);
 tanh.write(reinterpret_cast<char*>(t.data()),n*sizeof(float));sigmoid.write(reinterpret_cast<char*>(s.data()),n*sizeof(float));
#ifdef EIGEN_VECTORIZE_FMA
 std::cout<<"FMA enabled\n";
#else
 std::cout<<"FMA disabled\n";
#endif
}
