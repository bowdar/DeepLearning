# Meta-programming neural network 元编程神经网络

Compile-time matrix constructions 构造编译期矩阵以及数据传递代码  
headonly, no dependency, limitless layers, limitless nodes 纯头文件，无依赖，任意深度，允许超大结点数

Usage, the process is very simple 使用方法，过程极其简单
1) BPNN
```cpp
#include "RNN.hpp"

int main()
{
    /// 1. Create a neural network with an input layer two hide layers and an output layer 创建一个输入层，两个隐含层，一个输出层的神经网络 
    mtl::BPNN<20, 30, 20, 2> bpnn;

    /// 2. initialization 初始化
    bpnn.init(0.001, 0.8);

    /// 3. enter your matrix data... 录入你的矩阵数据
    mtl::Matrix<double, 1, 20> inMatrix;
    mtl::Matrix<double, 1, 2> outMatrix;
    ///    ... 

    /// 4. training 训练 
    bpnn.train(inMatrix, outMatrix, 100);

    /// 5. simuation 仿真
    bpnn.simulate(inMatrix, outMatrix);
```

2) RNN
```cpp
#include "RNN.hpp"

int main()
{
    /// 1. 创建一个输入层，两个隐含层，一个输出层的神经网络
    typedef mtl::RNN<20, 30, 20, 2> MyRnn;
    MyRnn rnn;
	
    /// 2. 初始化
    rnn.init(0.1, 0.8);

    /// 3. 输入，循环网络支持多输入多输出，InMatrix<10>表示10组输入
    MyRnn::InMatrix<10> inMatrix;
    MyRnn::OutMatrix<1> outMatrix;
    ///    录入你的矩阵数据...
	
    /// 4. 训练
    rnn.train(inMatrix, outMatrix, 100);
	
    /// 5. 仿真
    rnn.simulate(inMatrix, outMatrix);
}
```
