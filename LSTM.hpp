//-------------------------------------------------------------------------------
// @brief
//     Long Short-Term Memory deep neural network
//
// @author
//     Millhaus.Chen @time 2017/09/10 11:04
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

namespace lstm
{
    enum Gate : unsigned char
    {   f = 0,
        i,
        C,
        o,
    };

    /// Type helper
    template<typename I, int... Layers>
    struct Type;
    template<std::size_t... I, int... Layers>
    struct Type<std::index_sequence<I...>, Layers...>
    {
        typedef /// Weights type
        std::tuple<
                Matrix<
                        double,
                        UnpackInts<I, Layers...>::value,
                        UnpackInts<I + 1, Layers...>::value
                >[o]...
        > Weights;

        typedef /// Thresholds and deltas type
        std::tuple<
                Matrix<
                        double,
                        1,
                        UnpackInts<I + 1, Layers...>::value
                >[o]...
        > Thresholds;

        typedef /// RWeights type
        std::tuple<
                Matrix<
                        double,
                        UnpackInts<I, Layers...>::value,
                        UnpackInts<I, Layers...>::value
                >[o]...
        > RWeights;

        /// RCells type
        template<int R>
        using RCells = std::tuple<
                Matrix<
                        double,
                        1,
                        UnpackInts<I + 1, Layers...>::value
                >[R]...
        >;

        /// Temp states type
        template<int R>
        using Temps = std::tuple<
                Matrix<
                        double,
                        1,
                        UnpackInts<I + 1, Layers...>::value
                >[R][o]...
        >;
    };
}

/// The LSTM neural network class
template<int... Layers>
class LSTM : public NNParam
{
    static const int N = sizeof...(Layers);
    using expander = int[];

public:
    template<int R>
    using  InMatrix = Matrix<double,
                             R,
                             UnpackInts<0, Layers...>::value>;
    template<int R>
    using OutMatrix = Matrix<double,
                             R,
                             UnpackInts<N - 1, Layers...>::value>;

public:
    LSTM<Layers...>& init();

    template<class LX, class LY, class W, class T, class RW, class CY, class RC, class S>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold, RW& rWeight, CY& cellY,
                 RC& rCell, S& state, int t, int rIn);
    template<class LX, class W, class T, class DX, class DY, class GD, class RW, class S,
            class RD, class RC, class CY>
    void backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, GD& gDelta,
                 RW& rWeight, S& state, RD& rDelta, RC& rCell, CY& cellY, int t, int r, int rIn);

    template<class IN, class OUT, std::size_t... I>
    bool train(IN& input, OUT& output, int times, double nor, std::index_sequence<I...>);
    template<class IN, class OUT>
    bool train(IN& input, OUT& output, int times = 1, double nor = 1)
    {   return train(input, output, times, nor, std::make_index_sequence<N - 1>());
    }

    template<class IN, class OUT, std::size_t... I>
    double simulate(IN& input, OUT& output, OUT& expect, double nor, std::index_sequence<I...>);
    template<class IN, class OUT>
    double simulate(IN& input, OUT& output, OUT& expect, double nor = 1)
    {   return simulate(input, output, expect, nor, std::make_index_sequence<N - 1>());
    }

public:
    std::tuple<Matrix<double, 1, Layers>...> m_layers;
    std::tuple<Matrix<double, 1, Layers>...> m_cells; /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_deltas; /// Final delta of every output, redundance 1
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Weights m_weights;
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_thresholds;
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_gDeltas; /// All gates deltas, redundance 1
    typename lstm::Type<std::make_index_sequence<N>, Layers...>::RWeights m_rWeights;  /// redundance 1
};

}

#include "include/LSTM.inl"