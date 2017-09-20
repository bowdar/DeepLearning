//-------------------------------------------------------------------------------
// @brief
//     Long Short-Term Memory deep neural network
//
// @author
//     Millhaus.Chen @time 2017/09/10 11:04
//-------------------------------------------------------------------------------
#pragma once

#include "../math/sigfunc.h"
#include "../math/Matrix.hpp"
#include "../utility/UnpackArgs.hpp"
#include "../TupleTool.hpp"
#include <tuple>
#include <utility>

namespace mtl {

namespace lstm
{
    enum Gate : unsigned char
    {
        f = 0,
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

        typedef /// Thresholds type
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
    };
}

/// The LSTM neural network class
template<int... Layers>
class LSTM
{
    static const int N = sizeof...(Layers);
    using expander = int[];
public:
    using InMatrix = Matrix<double, 1, UnpackInts<0, Layers...>::value>;
    using OutMatrix = Matrix<double, 1, UnpackInts<N - 1, Layers...>::value>;

public:
    void init();

    template<class LX, class LY, class W, class T, class RLY, class RW, class CY, class RCY, class TP>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold, RLY& rLayerY, RW& rWeight, CY& cellY, RCY& rCellY, TP& t);
    template<class LX, class W, class T, class DX, class DY, class RLY, class RWX, class RWY, class RDX>
    void reverse(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, RLY& rLayerY,
                 RWX& rWeightX, RWY& rWeightY, RDX& rDeltaX);

    template<std::size_t... I>
    bool train(const InMatrix& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>);
    bool train(const InMatrix& input, const OutMatrix& output, int times = 1, double nor = 1)
    {
        return train(input, output, times, nor, std::make_index_sequence<N - 1>());
    }

    template<std::size_t... I>
    double simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>);
    double simulate(const InMatrix& input, OutMatrix& output, OutMatrix& expect, double nor = 1)
    {
        return simulate(input, output, expect, nor, std::make_index_sequence<N - 1>());
    }

public:
    std::tuple<Matrix<double, 1, Layers>...> m_layers;
    std::tuple<Matrix<double, 1, Layers>...> m_rLayers;
    std::tuple<Matrix<double, 1, Layers>...> m_cells; /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_rCells; /// redundance 1
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Weights m_weights;
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_thresholds;
    typename lstm::Type<std::make_index_sequence<N>, Layers...>::RWeights m_rWeights;  /// redundance 1
    typename lstm::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_temps;  ///  To reduce temporary matrix allocated on the stack in the process of calculation
    std::tuple<Matrix<double, 1, Layers>...> m_deltas; /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_rDeltas; /// redundance 1
    OutMatrix m_aberrmx;

public:
    double m_aberration = 0.001;
    double m_learnrate = 0.1;
};

}

#include "LSTM.inl"