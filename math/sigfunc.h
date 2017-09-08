//-------------------------------------------------------------------------------
// sigfunc.h
//
// @author
//     Millhaus.Chen @time 2017/08/02 14:35
//-------------------------------------------------------------------------------
#pragma once

#include <cmath>

namespace mtl {

inline double tansig(double val)
{
    return 2.0 / (1.0 + std::exp(-2.0 * val)) - 1.0;
}

inline double dtansig(double val)
{
    double tanh = tansig(val);
    return 1 - tanh * tanh;
}

inline double logsig(double val)
{
    return 1.0 / (1.0 + std::exp(-val));
}

inline double dlogsig(double val)
{
    double sigmoid = logsig(val);
    return sigmoid * (1.0 - sigmoid);
}

}