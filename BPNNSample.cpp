#include "BPNN.hpp"

int main()
{
    /// 1. 创建一个输入层，两个隐含层，一个输出层的神经网络
    mtl::BPNN<20, 30, 20, 2> bpnn;
	
    /// 2. 初始化
    bpnn.init(0.1, 0.8);

    /// 3. 输入
    mtl::Matrix<double, 1, 20> inMatrix;
    mtl::Matrix<double, 1, 2> outMatrix;
    ///    录入你的矩阵数据...
	
    /// 4. 训练
    bpnn.train(inMatrix, outMatrix, 100);
	
    /// 5. 仿真
    bpnn.simulate(inMatrix, outMatrix);
}

