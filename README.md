# BPNeuralNetwork

构造编译期矩阵以及数据传递代码，headonly，代码初步实现，功能还未完善

使用方法，过程及其简单
```cpp
/// 1. 创建一个输入层，两个隐含层，一个输出层的神经网络
mtl::BPNeuralNet<20, 30, 20, 2> mynn;

/// 2. 初始化
mynn.init(0.1, 0.8);

mtl::Matrix<double, 1, 20> inMatrix;
mtl::Matrix<double, 1, 2> outMatrix;

/// 3. ... 录入你的矩阵数据

/// 4. 训练
mynn.train(inMatrix, outMatrix, 100);

/// 5. 仿真
mynn.simulat(inMatrix, outMatrix);
```
