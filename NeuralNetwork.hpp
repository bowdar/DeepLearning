//-------------------------------------------------------------------------------
// NeuralNetwork.hpp
//
// @author
//     Millhaus.Chen @time 2017/07/28 15:16
//-------------------------------------------------------------------------------
#pragma once

#include "math/sigfunc.h"
#include "math/Matrix.hpp"
#include <tuple>
#include <utility>

namespace mtl {

/// Unpack ints from variadic template
/// The compile-time integer array, following RCInt is reverse of ints e.g.
///    G{5, 3, 2, 4, 2} the (0, G) is 2 and (4, G) is 5
template<int N, int... Tail>
struct RCInt;
template<int N, int Tail>
struct RCInt<N, Tail>
{
    enum { value = Tail };
};
template<int N, int Head, int... Tail>
struct RCInt<N, Head, Tail...>
{
    enum { value = (N == sizeof...(Tail)) ? Head : RCInt<N, Tail...>::value };
};
template<int N, int... Ints>
struct UnpackInts
{
    enum { value = RCInt<(int)sizeof...(Ints) - N - 1, Ints...>::value };
};

/// Type helper
template<typename I, int... Layers> struct NNType;
template<std::size_t... I, int... Layers>
struct NNType<std::index_sequence<I...>, Layers...>
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

/// The neural network class
template<int... Layers>
class BPNeuralNet
{
    using InMatrix = Matrix<double, 1, UnpackInts<0, Layers...>::value>;
    using OutMatrix = Matrix<double, 1, UnpackInts<sizeof...(Layers) - 1, Layers...>::value>;
public:
    void init(double aberration, double learnrate);

    template<class LX, class LY, class W, class T>
    void forward(LX& layerX, LY& layerY, W& weight, T& threshold);
    template<class LX, class LY, class W, class T, class DX, class DY>
    void reverse(LX& layerX, LY& layerY, W& weight, T& threshold, DX& deltaX, DY& deltaY);

    template<std::size_t... I>
    bool train(const InMatrix& input, OutMatrix& output, int times, std::index_sequence<I...>);
    bool train(const InMatrix& input, OutMatrix& output, int times = 1)
    {
        return train(input, output, times, std::make_index_sequence<sizeof...(Layers) - 1>());
    }

    template<std::size_t... I>
    void simulat(const InMatrix& input, OutMatrix& output, std::index_sequence<I...>);
    void simulat(const InMatrix& input, OutMatrix& output)
    {
        simulat(input, output, std::make_index_sequence<sizeof...(Layers) - 1>());
    }

public:
    std::tuple<Matrix<double, 1, Layers>...> m_layers;
    typename NNType<std::make_index_sequence<sizeof...(Layers) - 1>, Layers...>::Weights m_weights;
    typename NNType<std::make_index_sequence<sizeof...(Layers) - 1>, Layers...>::Thresholds m_thresholds;
    std::tuple<Matrix<double, 1, Layers>...> m_deltas;
    OutMatrix m_aberrmx;

protected:
    double m_aberration, m_learnrate;
};

}

#include "NeuralNetwork.inl"