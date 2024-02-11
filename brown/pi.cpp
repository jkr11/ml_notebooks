#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>
#include <math.h> 
#include <cmath>

auto approx(int N) {
    int n = 0;
    for(int i = 0; i < N; i++) {
        double x = static_cast<double>(rand())/RAND_MAX;
        double y = static_cast<double>(rand())/RAND_MAX;
        if(std::sqrt(pow(x,2) + pow(y,2)) <= 1) {
            n++;
        }
    }
    return static_cast<double>(n) /N * 4;
}

int main() {
    auto pi = approx(100000000);
    printf("%f\n", pi);
    return 0;
}