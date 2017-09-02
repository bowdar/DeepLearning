//-------------------------------------------------------------------------------
// UnpackArgs.hpp
//
// @author
//     Millhaus.Chen @time 2017/08/01 11:49
//-------------------------------------------------------------------------------
#pragma once

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