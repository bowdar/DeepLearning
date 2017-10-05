//-------------------------------------------------------------------------------
// @author
//     Millhaus.Chen @time 2017/09/10 11:03
//-------------------------------------------------------------------------------
namespace mtl {

template<int... Layers>
LSTM<Layers...>& LSTM<Layers...>::init()
{
    using namespace lstm;
    mtl::for_each(m_weights, [](auto& weight) mutable
    {   for(int g = 0; g < o; ++g) weight[g].random(0, 1);
    });
    mtl::for_each(m_thresholds, [](auto& threshold) mutable
    {   for(int g = 0; g < o; ++g) threshold[g].random(0, 1);
    });
    mtl::for_each(m_rWeights, [](auto& rWeight) mutable
    {   for(int g = 0; g < o; ++g) rWeight[g].random(0, 1);
    });

    return *this;
}

template<int... Layers>
template<class LX, class LY, class W, class T, class RW, class CY, class RC, class S>
void LSTM<Layers...>::forward(LX& layerX, LY& layerY, W& weight, T& threshold, RW& rWeight, CY& cellY,
             RC& rCell, S& state, int t, int rIn)
{
    using namespace lstm;

    /// formula : state_? = σ (∑[0,t]layerX × weight_? + threshold_?)
    auto formula = [&](unsigned char g, auto func)
    {   if(t < rIn) layerY.multiply(layerX, weight[g]);
        if(t > 0) layerY.mult_sum(state[t - 1][g], rWeight[g]);
        layerY += threshold[g];
        layerY.foreach(func);
    };

    formula(f, logsig);
    formula(i, logsig);
    formula(C, tansig);
    formula(o, logsig);

    /// cellY = t_f * rCellY + t_i * t_C;
    cellY.hadamard(state[t][f], rCell[t]);
    cellY.hadamard_sum(state[t][i], state[t][C]);
    rCell[t] = cellY;

    /// layerY = t_o * tanh(cellY);
    cellY.foreach(tansig);
    layerY.hadamard(state[t][o], cellY);

    state[t][o] = layerY;
};

template<int... Layers>
template<class LX, class W, class T, class DX, class DY, class GD, class RW, class S,
        class RD, class RC, class CY>
void LSTM<Layers...>::backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, GD& gDelta,
              RW& rWeight, S& state, RD& rDelta, RC& rCell, CY& cellY, int t, int r, int rIn)
{
    using namespace lstm;

    /// 计算各个门当前delta分量
    gDelta[o].hadamard(deltaY, state[t][o]); /// temp parameter
    auto tempCY = cellY.foreach_n(dtansig);
    gDelta[f].hadamard(gDelta[o], tempCY).hadamard(rCell[t]).hadamard(state[t][f].foreach_n(dlogsig));
    gDelta[i].hadamard(gDelta[o], tempCY).hadamard(state[t][C]).hadamard(state[t][i].foreach_n(dlogsig));
    gDelta[C].hadamard(gDelta[o], tempCY).hadamard(rCell[t]).hadamard(state[t][i]).hadamard(state[t][C].foreach_n([](auto e){ return 1 - e * e;}));
    gDelta[o].hadamard(deltaY, cellY.foreach(tansig)).hadamard(state[t][o].foreach_n(dlogsig));

    /// 倒序计算循环过程中产生的delta（时间方向上的delta）
    auto delta_cal = [&](unsigned char g, auto func)
    {
        if(t >= r - 1)
        {   /// 倒数第一个时刻
            rDelta[t][g] = gDelta[g];
            gDelta[g].hadamard(rDelta[t][g], state[t][g]);
        } else if(t >= rIn)
        {   /// 对应输出的时刻
            rDelta[t][g] = gDelta[g];
            rDelta[t][g].mult_trans_sum(rWeight[g], state[t + 1][g].foreach(func));
            gDelta[g].hadamard_sum(rDelta[t][g], state[t][g]);
            rDelta[t][g] += rDelta[t + 1][g];
        } else
        {   /// 对应输入的时刻
            rDelta[t][g].mult_trans(rWeight[g], state[t + 1][g].foreach(func));
            gDelta[g].hadamard_sum(rDelta[t][g], state[t][g]);
            rDelta[t][g] += rDelta[t + 1][g];
        }
    };
    delta_cal(f, dlogsig);
    delta_cal(i, dlogsig);
    delta_cal(C, dtansig);
    delta_cal(o, dlogsig);

    /// 当计算完循环中的第一个delta后开始修正权重、阈值，并在最后计算上一层最后一个delta（深度方向上的deltaX）
    if(t == 0)
    {   deltaX.constant(0);
        for(int g = 0; g < o; ++g)
        {   rWeight[g].adjustW(state[t][g], rDelta[t][g], m_learnrate);
            weight[g].adjustW(layerX, gDelta[g], m_learnrate);
            threshold[g].adjustT(gDelta[g], m_learnrate);
            deltaX.mult_trans_sum(weight[g], gDelta[g]);
        }
        deltaX.hadamard(layerX.foreach([](auto x)
                        {   auto sigX = logsig(x);
                            auto tanhX = tansig(x);
                            auto stX = sigX * tanhX;
                            auto dsigX = sigX * (1 - sigX);
                            auto tanhstX = tanh(stX);
                            /// 此处用到了链式求导法则和莱布尼茨公式
                            return dsigX * tanh(stX) + sigX * (1 - tanhstX * tanhstX)
                                       * (dsigX * tanhX + sigX * (1 - tanhX * tanhX));
                        }));
    }
};

template<int... Layers>
template<class IN, class OUT, std::size_t... I>
bool LSTM<Layers...>::train(IN& input, OUT& output, int times, double nor, std::index_sequence<I...>)
{
    using namespace lstm;
    /// lstm 需要创建临时矩阵，用来保存当前系列输入的states，也就是临时的layer集合还有delta集合和out集合
    const int r = input.Row() + output.Row();
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template RCells<r> rCells;
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
                               std::get<I + 1>(m_cells),
                               std::get<I>(rCells),
                               std::get<I>(states),
                               t, input.Row()),
                               0)...};
            /// 1.2 计算出的out依次赋给output的每一层
            if(t >= input.Row())
            {   trainOut.set(layerN, t - input.Row(), 0);
            }
        }
        /// 2. 判断误差
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
                                std::get<N - I - 2>(m_gDeltas),
                                std::get<N - I - 1>(m_rWeights),
                                std::get<N - I - 2>(states),
                                std::get<N - I - 2>(rDeltas),
                                std::get<N - I - 2>(rCells),
                                std::get<N - I - 1>(m_cells),
                                t, r, input.Row()),
                                0)...};
        }
    }
    return false;
}

template<int... Layers>
template<class IN, class OUT, std::size_t... I>
double LSTM<Layers...>::simulate(IN& input, OUT& output, OUT& expect, double nor, std::index_sequence<I...>)
{
    using namespace lstm;
    /// lstm 需要创建临时矩阵，用来保存当前系列输入的states，也就是临时的layer集合还有delta集合和out集合
    const int r = input.Row() + output.Row();
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template RCells<r> rCells;
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<r> states;
    typename Type<std::make_index_sequence<N - 1>, Layers...>::template Temps<r> deltas;
    OUT trainOut;
    OUT aberration;

    auto& layer0 = std::get<0>(m_layers);
    auto& layerN = std::get<N - 1>(m_layers);
    /// 1. 正向传播
    for(int t = 0; t < r; ++t)
    {   /// 1.1 依次取input的每一层作为当前输入层
        layer0.subset(input, t, 0);
        layer0.normalize(nor);
        expander {(forward(std::get<I>(m_layers),
                           std::get<I + 1>(m_layers),
                           std::get<I>(m_weights),
                           std::get<I>(m_thresholds),
                           std::get<I + 1>(m_rWeights),
                           std::get<I + 1>(m_cells),
                           std::get<I>(rCells),
                           std::get<I>(states),
                           t, input.Row()),
                0)...};
        /// 1.2 计算出的out依次赋给output的每一层
        if(t >= input.Row())
        {   trainOut.set(layerN, t - input.Row(), 0);
        }
    }

    /// 2. 判断误差
    double error = aberration.subtract(output, trainOut).squariance() / 2;

    return error;
}

}