# Meta-programming neural network 元编程神经网络

Compile-time matrix constructions 构造编译期矩阵以及数据传递代码  
headonly, no dependency, limitless layers, limitless nodes 纯头文件，无依赖，任意深度，允许超大结点数

Usage, the process is very simple 使用方法，过程极其简单
### 1) BPNN
```cpp
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
```

### 2) RNN
```cpp
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
```


### 3) LSTM

coding ...

### 4) CNN

coming soon ...

### 5) instance of MNIST

planning ...

### 6) Future

VGG, RCNN, GAN ...
