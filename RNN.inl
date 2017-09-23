//-------------------------------------------------------------------------------
// @author
//     Millhaus.Chen @time 2017/09/02 16:34
//-------------------------------------------------------------------------------
namespace mtl {

template<int... Layers>
void RNN<Layers...>::init()
{
    mtl::for_each(m_weights, [](auto& weight) mutable
    {
        weight.random(0, 1);
    });

    mtl::for_each(m_thresholds, [](auto& threshold) mutable
    {
        threshold.random(0, 1);
    });

    mtl::for_each(m_rWeights, [](auto& rWeight) mutable
    {
        rWeight.random(0, 1);
    });

    std::get<N - 2>(m_rDeltas).constant(0);
}

template<int... Layers>
template<class LX, class LY, class W, class T, class RW, class S>
void RNN<Layers...>::forward(LX& layerX, LY& layerY, W& weight, T& threshold, RW& rWeight, S& state, int t)
{
    layerY.multiply(layerX, weight);
    if(t > 0) layerY.mult_sum(state[t - 1], rWeight); /// 循环中的第一个不累加上一个时刻的状态
    //layerY.foreach([&layerX](auto& e){ return e / layerX.Col();});
    layerY += threshold;
    layerY.foreach(logsig);
    state[t] = layerY;
};

template<int... Layers>
template<class LX, class W, class T, class DX, class DY, class RW, class S, class DT>
void RNN<Layers...>::reverse(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY,
                             RW& rWeight, S& state, DT& delta, int t, int r)
{
    /// 倒序计算循环过程中产生的delta
    if(t >= r - 1)
    {
        delta[t] = deltaY;
        deltaY.hadamard(delta[t], state[t]);
    }
    else
    {
        delta[t].mult_trans(rWeight, state[t + 1].foreach(dlogsig));
        delta[t].hadamard(delta[t + 1]);
        deltaY.hadamard_sum(delta[t], state[t]);
    }

    /// 当计算完循环中的第一个delta后开始修正权重、阈值和上一层最后一个delta
    if(t == 0)
    {
        weight.adjustW(layerX, deltaY, m_learnrate);
        threshold.adjustT(deltaY, m_learnrate);

        deltaX.mult_trans(weight, deltaY);
        deltaX.hadamard(layerX.foreach(dlogsig));
    }
};

template<int... Layers>
template<class InMx, std::size_t... I>
bool RNN<Layers...>::train(InMx& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>)
{
    /// rnn 需要创建临时矩阵，用来保存当前系列输入的states，也就是临时的layer集合还有delta集合
    typename rnn::Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<input.Row()> states;
    typename rnn::Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<input.Row()> deltas;

    auto& layerN = std::get<N - 1>(m_layers);
    auto& deltaN = std::get<N - 1>(m_deltas);
    for(int i = 0; i < times; ++i)
    {
        auto& layer0 = std::get<0>(m_layers);
        for(int t = 0; t < input.Row(); ++t)
        {
            /// 1. 依次取input的每一层作为当前输入层
            layer0 = input.row(t);
            /// 2. 输入归一化
            layer0.normalize(nor);
            /// 3. 正向传播
            expander {(forward(std::get<I>(m_layers),
                               std::get<I + 1>(m_layers),
                               std::get<I>(m_weights),
                               std::get<I>(m_thresholds),
                               std::get<I + 1>(m_rWeights),
                               std::get<I>(states),
                               t),
                               0)...};
        }
        /// 4. 判断误差
        double aberration = m_aberrmx.subtract(output, layerN).squariance() / 2;
        if (aberration < m_aberration) break;
        /// 5. 反向修正
        deltaN.hadamard(m_aberrmx, layerN.foreach(dlogsig));
        for(int t = input.Row() - 1; t >= 0; --t)
        {
            expander {(reverse(std::get<N - I - 2>(m_layers),
                               std::get<N - I - 2>(m_weights),
                               std::get<N - I - 2>(m_thresholds),
                               std::get<N - I - 2>(m_deltas),
                               std::get<N - I - 1>(m_deltas),
                               std::get<N - I - 1>(m_rWeights),
                               std::get<N - I - 2>(states),
                               std::get<N - I - 2>(deltas),
                               t, input.Row()),
                               0)...};
        }
    }
    return false;
}

template<int... Layers>
template<class InMx, std::size_t... I>
double RNN<Layers...>::simulate(const InMx& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>)
{
    /// 1. 输入归一化
    auto& layer0 = std::get<0>(m_layers);
    layer0 = input;
    layer0.normalize(nor);
    /// 2. 正向传播
    expander {(forward(std::get<I>(m_layers),
                       std::get<I + 1>(m_layers),
                       std::get<I>(m_weights),
                       std::get<I>(m_thresholds)),
                       0)...};
    /// 3. 输出结果
    output = std::get<N - 1>(m_layers);

    /// 4. 判断误差
    double aberration = m_aberrmx.subtract(expect, output).squariance() / 2;

    return aberration;
}

}