#pragma once

#include <tuple>

namespace mtl
{

//-------------------------------------------------------------------------------
// @description
//     代替 c++14 才有的std::get<Type>
//
// @added
//     Millhaus.Chen @time 2015/10/29 13:20
//
// @modified
//     增加get返回默认值的功能 Millhaus.Chen @time 2016/03/14 14:37
//-------------------------------------------------------------------------------
template<size_t N, typename T, typename ...Args>
struct GetType;

template<size_t N, typename T, typename Head, typename ...Args>
struct GetType<N, T, Head, Args...>
{
    enum
    {
        value = std::is_same<T, Head>::value ? N : GetType<N + 1, T, Args...>::value
    };
};

template<size_t N, typename T>
struct GetType<N, T>
{
    enum
    {
        value = 0
    };
};

template<typename T, typename ...Args>
T get(std::tuple<Args...> const &tuple, T &&def_val = T{})
{
    return std::get<GetType<0, T, Args..., T>::value>(
            std::tuple_cat(tuple, std::make_tuple(def_val)));
}
//template<typename T, typename ...Args>
//T get(std::tuple<Args...> const& tuple)
//{
//	return std::get<GetType<0, T, Args...>::value>(tuple);
//}


//-------------------------------------------------------------------------------
// @description
//     tuple_for_each
//
// @added
//     Millhaus.Chen @time 2017/03/16 10:21
//-------------------------------------------------------------------------------
template<class F, class... Ts, std::size_t... Is>
void for_each(std::tuple<Ts...>& tuple, F func, std::index_sequence<Is...>)
{
    using expander = int[];
    (void) expander {0, ((void) func(std::get<Is>(tuple)), 0)...};
}

template<class F, class... Ts>
void for_each(std::tuple<Ts...>& tuple, F func)
{
    for_each(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

}