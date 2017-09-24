//-------------------------------------------------------------------------------
// @author
//     Millhaus.Chen @time 2017/09/10 11:03
//-------------------------------------------------------------------------------
namespace mtl {

template<int... Layers>
void LSTM<Layers...>::init()
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
    std::get<0>(m_rLayers).constant(0);
    std::get<0>(m_rCells).constant(0);
    std::get<N - 2>(m_rDeltas).constant(0);
}

template<int... Layers>
template<class LX, class LY, class W, class T, class RLY, class RW, class CY, class RCY, class TP>
void LSTM<Layers...>::forward(LX& layerX, LY& layerY, W& weight, T& threshold, RLY& rLayerY, RW& rWeight,
                              CY& cellY, RCY& rCellY, TP& t)
{
    using namespace lstm;

    auto t_func = [&](unsigned char g, auto func)
    {   /// t_? = σ (layerX × weight_? + rLayerY × weight_? + threshold_?)
        t[g].multiply(layerX, weight[g]);
        t[g].mult_sum(rLayerY, rWeight[g]);
        t[g] += threshold[g];
        t[g].foreach(func);
    };
    t_func(f, logsig);
    t_func(i, logsig);
    t_func(C, tansig);
    t_func(o, logsig);

    /// cellY = t_f * rCellY + t_i * t_C;
    cellY.hadamard(t[f], rCellY);
    cellY.hadamard_sum(t[i], t[C]);
    rCellY = cellY;

    /// layerY = t_o * tanh(cellY);
    cellY.foreach(tansig);
    layerY.hadamard(t[o], cellY);

    rLayerY = layerY;
};

template<int... Layers>
template<class LX, class W, class T, class DX, class DY, class RLY, class RWX, class RWY, class RDX>
void LSTM<Layers...>::reverse(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, RLY& rLayerY,
                              RWX& rWeightX, RWY& rWeightY, RDX& rDeltaX)
{
//    weight.adjustW(layerX, deltaY, m_learnrate);
//    rWeightY.adjustW(rLayerY, deltaY, m_learnrate);
//    threshold.adjustT(deltaY, m_learnrate);
//    /// 计算delta
//    deltaX.multtrans(weight, deltaY);
//    deltaX.multtrans(rWeightX, rDeltaX);
//    deltaX.hadamard(layerX, layerX.foreach(dlogsig));
//    rDeltaX = deltaX;
};

template<int... Layers>
template<std::size_t... I>
bool LSTM<Layers...>::train(const InMatrix& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>)
{
    /// 1. 输入归一化
    auto& layer0 = std::get<0>(m_layers);
    layer0 = input;
    layer0.normalize(nor);
    auto& layerN = std::get<N - 1>(m_layers);
    auto& deltaN = std::get<N - 1>(m_deltas);
    for(int i = 0; i < times; ++i)
    {   /// 2. 正向传播
        expander {(forward(std::get<I>(m_layers),
                           std::get<I + 1>(m_layers),
                           std::get<I>(m_weights),
                           std::get<I>(m_thresholds),
                           std::get<I + 1>(m_rLayers),
                           std::get<I + 1>(m_rWeights),
                           std::get<I + 1>(m_cells),
                           std::get<I + 1>(m_rCells),
                           std::get<I>(m_temps)),
                           0)...};
        /// 3. 判断误差
        double aberration = m_aberrmx.subtract(output, layerN).squariance() / 2;
        if (aberration < m_aberration) break;
//        /// 4. 反向修正
//        deltaN.hadamard(m_aberrmx, layerN.foreach(dlogsig));
//        expander {(reverse(std::get<N - I - 2>(m_layers),
//                           std::get<N - I - 2>(m_weights),
//                           std::get<N - I - 2>(m_thresholds),
//                           std::get<N - I - 2>(m_deltas),
//                           std::get<N - I - 1>(m_deltas),
//                           std::get<N - I - 1>(m_rLayers),
//                           std::get<N - I - 2>(m_rDeltas)),
//                           0)...};
    }
    return false;
}

template<int... Layers>
template<std::size_t... I>
double LSTM<Layers...>::simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>)
{
    /// 1. 输入归一化
    auto& layer0 = std::get<0>(m_layers);
    layer0 = input;
    layer0.normalize(nor);
    /// 2. 正向传播
    expander {(forward(std::get<I>(m_layers),
                       std::get<I + 1>(m_layers),
                       std::get<I>(m_weights),
                       std::get<I>(m_thresholds),
                       std::get<I + 1>(m_rLayers),
                       std::get<I + 1>(m_rWeights),
                       std::get<I + 1>(m_cells),
                       std::get<I + 1>(m_rCells),
                       std::get<I>(m_temps)),
                       0)...};
    /// 3. 输出结果
    output = std::get<N - 1>(m_layers);

    /// 4. 判断误差
    double aberration = m_aberrmx.subtract(expect, output).squariance() / 2;

    return aberration;
}

}