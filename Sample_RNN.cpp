#include "RNN.hpp"

int main()
{
    /// 1. 创建一个输入层，两个隐含层，一个输出层的神经网络
    typedef mtl::RNN<20, 30, 20, 2> MyRnn;
	MyRnn rnn;
	
    /// 2. 初始化
    rnn.init(0.1, 0.8);

    /// 3. 输入，循环网络支持多输入多输出，InMatrix<10>表示10组输入
    MyRnn::InMatrix<10> inMx;
    MyRnn::OutMatrix<1> outMx;
	MyRnn::OutMatrix<1> expectMx;
    ///    录入你的矩阵数据...
	
    /// 4. 训练
    rnn.train(inMx, outMx, 100);
	
    /// 5. 仿真
    rnn.simulate(inMx, outMx，expectMx);
}

