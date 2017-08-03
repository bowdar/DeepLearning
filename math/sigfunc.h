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
    return 1 / (std::atan(val) + 1);
}

inline double logsig(double val)
{
    return 1 / (1 + std::exp(-val));
}

inline double dlogsig(double val)
{
    return val * (1 - val);
}

}