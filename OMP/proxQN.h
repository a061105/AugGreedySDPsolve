#ifndef PROXQN
#define PROXQN
#include"optimizer.h"

class proxQN:public optimizer{
    public:
//        proxQN();
//        ~proxQN();
        void minimize(Problem *prob);
}; 

#endif
