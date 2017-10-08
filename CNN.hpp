//-------------------------------------------------------------------------------
// @brief
//     Deep CNN neural network
//
// @author
//     Millhaus.Chen @time 2017/10/07 10:58
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

namespace cnn
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
    };
}

/// The neural network class
template<int... Layers>
class CNN : public NNParam
{
    static const int N = sizeof...(Layers);
    using expander = int[];

public:
    using InMatrix = Matrix<double, 1, UnpackInts<0, Layers...>::value>;
    using OutMatrix = Matrix<double, 1, UnpackInts<N - 1, Layers...>::value>;

public:
    CNN<Layers...>& init();

    template<class LX, class LY, class W, class T>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold);
    template<class LX, class W, class T, class DX, class DY>
    void backward(LX& layerX, W& weight, T& threshold, DX& deltaX, DY& deltaY);

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
    typename cnn::Type<std::make_index_sequence<N - 1>, Layers...>::Weights m_weights;
    typename cnn::Type<std::make_index_sequence<N - 1>, Layers...>::Thresholds m_thresholds;
    std::tuple<Matrix<double, 1, Layers>...> m_deltas;
    OutMatrix m_aberrmx;
};

}

#include "include/CNN.inl"