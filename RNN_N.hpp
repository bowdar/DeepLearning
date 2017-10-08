//-------------------------------------------------------------------------------
// @brief
//     Recurrent deep neural network
//     Different from RNN, the in and out of RNN_N is all in circulate and only
//  support 1:1 mode
//
// @author
//     Millhaus.Chen @time 2017/09/02 16:34
//-------------------------------------------------------------------------------
#pragma once

#include "math/sigfunc.h"
#include "math/Matrix.hpp"
#include "util/UnpackArgs.hpp"
#include "util/TupleTool.hpp"
#include "include/Parameter.hpp"

#include <tuple>
#include <utility>

namespace mtl {

/// Type helper
template<typename I, int... Layers> struct RNNType;
template<std::size_t... I, int... Layers>
struct RNNType<std::index_sequence<I...>, Layers...>
{
    typedef /// Weights type
    std::tuple<
            Matrix<
                    double,
                    UnpackInts<I, Layers...>::value,
                    UnpackInts<I + 1, Layers...>::value
            >...
    > Weights;

    typedef /// Thresholds type
    std::tuple<
            Matrix<
                    double,
                    1,
                    UnpackInts<I + 1, Layers...>::value
            >...
    > Thresholds;

    typedef /// RWeights type
    std::tuple<
            Matrix<
                    double,
                    UnpackInts<I, Layers...>::value,
                    UnpackInts<I, Layers...>::value
            >...
    > RWeights;
};

/// The neural network class
template<int... Layers>
class RNN_N : NNParam
{
    static const int N = sizeof...(Layers);
    using expander = int[];
public:
    using InMatrix = Matrix<double, 1, UnpackInts<0, Layers...>::value>;
    using OutMatrix = Matrix<double, 1, UnpackInts<N - 1, Layers...>::value>;

public:
    RNN_N<Layers...>& init();

    template<class LX, class LY, class W, class T, class RLY, class RW>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold, RLY& rLayerY, RW& rWeight);
    template<class LX, class W, class T, class DX, class DY, class RLY, class RWX, class RWY, class RDX>
    void backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, RLY& rLayerY,
                 RWX& rWeightX, RWY& rWeightY, RDX& rDeltaX);

    template<std::size_t... I>
    bool train(const InMatrix& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>);
    bool train(const InMatrix& input, const OutMatrix& output, int times = 1, double nor = 1)
    {   return train(input, output, times, nor, std::make_index_sequence<N - 1>());
    }

    template<std::size_t... I>
    double simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>);
    double simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor = 1)
    {   return simulate(input, output, expect, nor, std::make_index_sequence<N - 1>());
    }

public:
    std::tuple<Matrix<double, 1, Layers>...> m_layers;
    std::tuple<Matrix<double, 1, Layers>...> m_rLayers;
    typename RNNType<std::make_index_sequence<N - 1>, Layers...>::Weights m_weights;
    typename RNNType<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_thresholds;
    typename RNNType<std::make_index_sequence<N>, Layers...>::RWeights m_rWeights;  /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_deltas; /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_rDeltas; /// redundance 1
    OutMatrix m_aberrmx;
};

}

#include "include/RNN_N.inl"