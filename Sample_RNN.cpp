#include "RNN.hpp"

int main()
{
    /// 1. Create a 4 layers NN each layer nodes are 20, 30, 20 and 2
    ///    The first 20 is input layer and the last 2 is output
    typedef mtl::RNN<20, 30, 20, 2> MyRnn;
    MyRnn rnn;
    
    /// 2. Initialize, setup parameters and activate functions
    bpnn.init()
        .set_aberration(0.0001)
        .set_learnrate(0.8)
        .set_sigfunc(mtl::logsig)
        .set_dsigfunc(mtl::dlogsig);

    /// 3. Create input output matrixs, and then enter matrix datas your self
    ///    RNN suport multi-in-out like M:1, 1:M and M:M also 1:1 which is meaningless
    MyRnn::InMatrix<10> inMx; /// 10 input a group, you can change it each training
    MyRnn::OutMatrix<2> outMx; /// 2 ouput a group
    MyRnn::OutMatrix<2> expectMx;
    ///    enter matrix datas ...
    
    /// 4. Training, call train in your own way
    rnn.train(inMx, outMx, 100);
    
    /// 5. Simulate
    rnn.simulate(inMx, outMx，expectMx);
}

