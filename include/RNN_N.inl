//-------------------------------------------------------------------------------
// @author
//     Millhaus.Chen @time 2017/09/02 16:34
//-------------------------------------------------------------------------------
namespace mtl {

template<int... Layers>
RNN_N<Layers...>& RNN_N<Layers...>::init()
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
    std::get<0>(m_rLayers).constant(0);
    std::get<N - 2>(m_rDeltas).constant(0);
	
	return *this;
}

template<int... Layers>
template<class LX, class LY, class W, class T, class RLY, class RW>
void RNN_N<Layers...>::forward(LX& layerX, LY& layerY, W& weight, T& threshold, RLY& rLayerY, RW& rWeight)
{
    layerY.multiply(layerX, weight); /// layerY = layerX * weight
    layerY.mult_sum(rLayerY, rWeight);
    layerY.foreach([&layerX](auto& e){ return e / layerX.Col();});
    layerY += threshold;
    layerY.foreach(m_sigfunc);
    rLayerY = layerY;
};

template<int... Layers>
template<class LX, class W, class T, class DX, class DY, class RLY, class RWX, class RWY, class RDX>
void RNN_N<Layers...>::backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, RLY& rLayerY,
                             RWX& rWeightX, RWY& rWeightY, RDX& rDeltaX)
{
    weight.adjustW(layerX, deltaY, m_learnrate);
    rWeightY.adjustW(rLayerY, deltaY, m_learnrate);
    threshold.adjustT(deltaY, m_learnrate);
    /// 计算delta
    deltaX.mult_trans(weight, deltaY);
    deltaX.mult_trans_sum(rWeightX, rDeltaX);
    layerX.foreach(m_dsigfunc);
    deltaX.hadamard(layerX);
    rDeltaX = deltaX;
};

template<int... Layers>
template<std::size_t... I>
bool RNN_N<Layers...>::train(const InMatrix& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>)
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
                           std::get<I + 1>(m_rWeights)),
                0)...};
        /// 3. 判断误差
        double aberration = m_aberrmx.subtract(output, layerN).squariance() / 2;
        if (aberration < m_aberration) break;
        /// 4. 反向修正
        deltaN.hadamard(m_aberrmx, layerN.foreach(dlogsig));
        expander {(backward(std::get<N - I - 2>(m_layers),
                           std::get<N - I - 2>(m_weights),
                           std::get<N - I - 2>(m_thresholds),
                           std::get<N - I - 2>(m_deltas),
                           std::get<N - I - 1>(m_deltas),
                           std::get<N - I - 1>(m_rLayers),
                           std::get<N - I - 2>(m_rWeights),
                           std::get<N - I - 1>(m_rWeights),
                           std::get<N - I - 2>(m_rDeltas)),
                0)...};
    }
    return false;
}

template<int... Layers>
template<std::size_t... I>
double RNN_N<Layers...>::simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>)
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