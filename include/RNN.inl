//-------------------------------------------------------------------------------
// @author
//     Millhaus.Chen @time 2017/09/02 16:34
//-------------------------------------------------------------------------------
namespace mtl {

template<int... Layers>
RNN<Layers...>& RNN<Layers...>::init()
{
    mtl::for_each(m_weights, [](auto& weight) mutable
    {   weight.random(0, 1);
    });
    mtl::for_each(m_thresholds, [](auto& threshold) mutable
    {   threshold.random(0, 1);
    });
    mtl::for_each(m_rWeights, [](auto& rWeight) mutable
    {   rWeight.random(0, 1);
    });

    return *this;
}

template<int... Layers>
template<class LX, class LY, class W, class T, class RW, class S>
void RNN<Layers...>::forward(LX& layerX, LY& layerY, W& weight, T& threshold, RW& rWeight, S& state, int t, int rIn)
{
    if(t < rIn) layerY.multiply(layerX, weight); /// 循环中只有输入部分才计算输入层
    if(t > 0) layerY.mult_sum(state[t - 1], rWeight); /// 循环中的第一个不累加上一个时刻的状态
    layerY += threshold;
    layerY.foreach(m_sigfunc);
    state[t] = layerY;
};

template<int... Layers>
template<class LX, class W, class T, class DX, class DY, class RW, class S, class RD>
void RNN<Layers...>::backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY,
                             RW& rWeight, S& state, RD& rDelta, int t, int r, int rIn)
{
    /// 倒序计算循环过程中产生的delta（时间方向上的delta）
    if(t >= r - 1)
    {   /// 倒数第一个时刻
        rDelta[t] = deltaY;
        deltaY.hadamard(rDelta[t], state[t]);
    } else if(t >= rIn)
    {   /// 对应输出的时刻
        rDelta[t] = deltaY;
        rDelta[t].mult_trans_sum(rWeight, state[t + 1].foreach(m_dsigfunc));
        deltaY.hadamard_sum(rDelta[t], state[t]);
        rDelta[t] += rDelta[t + 1];
    } else
    {   /// 对应输入的时刻
        rDelta[t].mult_trans(rWeight, state[t + 1].foreach(m_dsigfunc));
        deltaY.hadamard_sum(rDelta[t], state[t]);
        rDelta[t] += rDelta[t + 1];
    }
    /// 当计算完循环中的第一个delta后开始修正权重、阈值
    if(t == 0)
    {   rWeight.adjustW(state[t], rDelta[t], m_learnrate);
        weight.adjustW(layerX, deltaY, m_learnrate);
        threshold.adjustT(deltaY, m_learnrate);
        /// 计算上一层最后一个delta（深度方向上的deltaX）
        deltaX.mult_trans(weight, deltaY);
        deltaX.hadamard(layerX.foreach(m_dsigfunc));
    }
};

template<int... Layers>
template<class IN, class OUT, std::size_t... I>
bool RNN<Layers...>::train(IN& input, OUT& output, int times, double nor, std::index_sequence<I...>)
{
    using namespace rnn;
    /// rnn 需要创建临时矩阵，用来保存当前系列输入的states，也就是临时的layer集合还有delta集合和out集合
    const int r = input.Row() + output.Row();
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<r> states;
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<r> rDeltas;
    OUT trainOut;
    OUT aberration;

    auto& layer0 = std::get<0>(m_layers);
    auto& layerN = std::get<N - 1>(m_layers);
    auto& deltaN = std::get<N - 1>(m_deltas);
    for(int i = 0; i < times; ++i)
    {   /// 1. 正向传播
        for(int t = 0; t < r; ++t)
        {   /// 1.1 依次取input的每一层作为当前输入层
            layer0.subset(input, t, 0);
            layer0.normalize(nor);
            expander {(forward(std::get<I>(m_layers),
                               std::get<I + 1>(m_layers),
                               std::get<I>(m_weights),
                               std::get<I>(m_thresholds),
                               std::get<I + 1>(m_rWeights),
                               std::get<I>(states),
                               t, input.Row()),
                               0)...};
            /// 1.2 计算出的out依次赋给output的每一层
            if(t >= input.Row())
            {   trainOut.set(layerN, t - input.Row(), 0);
            }
        }
        /// 2. 判断误差，这里是多个输出的总误差
        double error = aberration.subtract(output, trainOut).squariance() / 2;
        if (error < m_aberration) break;
        /// 3. 反向修正
        for(int t = r - 1; t >= 0; --t)
        {   if(t > input.Row() - 1)
            {   deltaN.subset(aberration, t - input.Row(), 0);
                deltaN.hadamard(layerN.foreach(dlogsig));
            }
            expander {(backward(std::get<N - I - 2>(m_layers),
                                std::get<N - I - 2>(m_weights),
                                std::get<N - I - 2>(m_thresholds),
                                std::get<N - I - 2>(m_deltas),
                                std::get<N - I - 1>(m_deltas),
                                std::get<N - I - 1>(m_rWeights),
                                std::get<N - I - 2>(states),
                                std::get<N - I - 2>(rDeltas),
                                t, r, input.Row()),
                                0)...};
        }
    }
    return false;
}

template<int... Layers>
template<class IN, class OUT, std::size_t... I>
double RNN<Layers...>::simulate(IN& input, OUT& output, OUT& expect, double nor, std::index_sequence<I...>)
{
    const int r = input.Row() + output.Row();
    typename rnn::Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<r> rStates;

    /// 正向传播
    auto& layer0 = std::get<0>(m_layers);
    auto& layerN = std::get<N - 1>(m_layers);
    for(int t = 0; t < r; ++t)
    {   input.subset(layer0, t, 0);
        layer0.normalize(nor);
        expander {(forward(std::get<I>(m_layers),
                           std::get<I + 1>(m_layers),
                           std::get<I>(m_weights),
                           std::get<I>(m_thresholds),
                           std::get<I + 1>(m_rWeights),
                           std::get<I>(rStates),
                           t, input.Row()),
                0)...};
        if(t >= input.Row())
        {   output.set(layerN, t - input.Row(), 0);
        }
    }

    /// 返回误差
    return output.subtract(expect, output).squariance() / 2;
}

}