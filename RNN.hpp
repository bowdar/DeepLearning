//-------------------------------------------------------------------------------
// @brief
//     Recursion deep neural network
//
// @author
//     Millhaus.Chen @time 2017/09/02 16:34
//-------------------------------------------------------------------------------
#pragma once

#include "math/sigfunc.h"
#include "math/Matrix.hpp"
#include "util/UnpackArgs.hpp"
#include "util/TupleTool.hpp"
#include <tuple>
#include <utility>

namespace mtl {

namespace rnn
{
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

        /// Temps type
        template<int R>
        using Temps =
        std::tuple<
                Matrix<
                        double,
                        1,
                        UnpackInts<I + 1, Layers...>::value
                >[R]...
        >;
    };
}
/// The neural network class
template<int... Layers>
class RNN
{
    static const int N = sizeof...(Layers);
    using expander = int[];

public:
    template<int R>
    using InMatrix = Matrix<double,
                            R,
                            UnpackInts<0, Layers...>::value>;

    using OutMatrix = Matrix<double,
                             1,
                             UnpackInts<N - 1, Layers...>::value>;

public:
    void init();

    template<class LX, class LY, class W, class T, class RW, class S>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold, RW& rWeight, S& state, int t);
    template<class LX, class W, class T, class DX, class DY, class RW, class S, class DT>
    void reverse(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY, RW& rWeight,
                 S& state, DT& delta, int t, int r);

    template<class InMx, std::size_t... I>
    bool train(InMx& input, const OutMatrix& output, int times, double nor, std::index_sequence<I...>);
    template<class InMx>
    bool train(InMx& input, const OutMatrix& output, int times = 1, double nor = 1)
    {
        return train(input, output, times, nor, std::make_index_sequence<N - 1>());
    }

    template<class InMx, std::size_t... I>
    double simulate(const InMx& input, OutMatrix& output, OutMatrix& expect, double nor, std::index_sequence<I...>);
    template<class InMx>
    double simulate(const InMx& input, OutMatrix& output, OutMatrix& expect, double nor = 1)
    {
        return simulate(input, output, expect, nor, std::make_index_sequence<N - 1>());
    }

public:
    std::tuple<Matrix<double, 1, Layers>...> m_layers;
    typename rnn::Type<std::make_index_sequence<N - 1>, Layers...>::Weights m_weights;
    typename rnn::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_thresholds;
    typename rnn::Type<std::make_index_sequence<N>, Layers...>::RWeights m_rWeights;  /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_deltas; /// redundance 1
    std::tuple<Matrix<double, 1, Layers>...> m_rDeltas; /// redundance 1
    OutMatrix m_aberrmx;

public:
    double m_aberration = 0.001;
    double m_learnrate = 0.1;
};

}

#include "RNN.inl"