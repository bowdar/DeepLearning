#include "BPNN.hpp"

int main()
{
    /// 1. Create a 4 layers NN each layer nodes are 20, 30, 20 and 2
	///    The first 20 is input layer and the last 2 is output
    typedef mtl::BPNN<20, 30, 20, 2> MyNN;
	MyNN bpnn;
	
	/// 2. Initialize, setup parameters and activate functions
	bpnn.init()
	    .set_aberration(0.0001)
        .set_learnrate(0.8)
        .set_sigfunc(mtl::logsig)
        .set_dsigfunc(mtl::dlogsig);

    /// 3. Create input output matrixs, and then enter matrix datas your self
    MyNN::InMatrix inMx;
    MyNN::OutMatrix outMx;
	MyNN::OutMatrix expectMx;
    ///    enter matrix datas ...
	
    /// 4. Training, call train in your own way
    bpnn.train(inMx, outMx, 100);
	
    /// 5. Simulate
    bpnn.simulate(inMx, outMx, expectMx);
}

