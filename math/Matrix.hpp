//-------------------------------------------------------------------------------
// Matrix.hpp
//
// @brief
//     thread-unsafe
//
// @author
//     Millhaus.Chen @time 2017/07/27 14:21
//-------------------------------------------------------------------------------
#pragma once

#include <cstdlib>
#include <cmath>

namespace mtl {

template<typename DataType, int ROW, int COL>
class Matrix
{
public:
    /// 提供两个遍历函数，第一个会遍历并修改每个元素
    template<typename F>
    Matrix& foreach(F func)
    {
        for(auto& r : data)
        {
            for(auto& e : r)
            {
                e = func(e);
            }
        }
        return *this;
    }
    template<typename F>
    Matrix& foreach_c(F func)
    {
        for(auto& r : data)
        {
            for(auto& e : r)
            {
                func(e);
            }
        }
        return *this;
    }

    /// 平方和
    DataType squariance()
    {
        DataType ret = 0;
        foreach_c([&ret](auto& e){ ret += e * e; });
        return ret;
    }

    /// 归一化
    void normaliz1()
    {
        DataType sqc = squariance();
        if (sqc == 0) return;
        foreach_c([&sqc](auto& e){ e = e / sqc; });
    }

    /// 矩阵乘法
    template<int RC>
    Matrix& multiply(const Matrix<DataType, ROW, RC>& mX, const Matrix<DataType, RC, COL>& mY)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] = 0;
                for (int k = 0; k < RC; ++k)
                {
                    data[i][j] += mX.data[i][k] * mY.data[k][j];
                }
            }
        }
        return *this;
    }
    /// 矩阵乘法带转置
    template<int RC>
    Matrix& multtrans(const Matrix<DataType, COL, RC>& mX, const Matrix<DataType, ROW, RC>& mY)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] = 0;
                for (int k = 0; k < RC; ++k)
                {
                    data[i][j] += mX.data[j][k] * mY.data[i][k];
                }
            }
        }
        return *this;
    }
    /// Matrix(ROW, COL) * Matrix(COL, COL_A)
    template<int COL_A>
    Matrix<DataType, ROW, COL_A> operator*(const Matrix<DataType, COL, COL_A>& m) const
    {
        Matrix<DataType, ROW, COL_A> ret;
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL_A; ++j)
            {
                ret.data[i][j] = 0;
                for (int k = 0; k < COL; ++k)
                {
                    ret.data[i][j] += data[i][k] * m.data[k][j];
                }
            }
        }
        return ret;
    }

    Matrix operator+(const Matrix& m) const
    {
        Matrix ret;
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                ret.data[i][j] = data[i][j] + m.data[i][j];
            }
        }
        return ret;
    }
    Matrix& operator+=(const Matrix& m)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] += m.data[i][j];
            }
        }
        return *this;
    }

    Matrix& subtract(const Matrix& mX, const Matrix& mY)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] = mX.data[i][j] - mY.data[i][j];
            }
        }
        return *this;
    }
    Matrix operator-(const Matrix& m) const
    {
        Matrix ret;
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                ret.data[i][j] = data[i][j] - m.data[i][j];
            }
        }
        return ret;
    }
    Matrix& operator-=(const Matrix& m)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] -= m.data[i][j];
            }
        }
        return *this;
    }

    /// 转置
    Matrix<DataType, COL, ROW> transpose() const
    {
        Matrix<DataType, COL, ROW> ret;
        for (int i = 0; i < COL; ++i)
        {
            for (int j = 0; j < ROW; ++j)
            {
                ret.data[i][j] = data[j][i];
            }
        }
        return ret;
    }

    /// Hadamard product 哈达玛积
    Matrix& hadamard(const Matrix& m)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] *= m.data[i][j];
            }
        }
        return *this;
    }
    Matrix& hadamard(const Matrix& mX, const Matrix& mY)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] = mX.data[i][j] * mY.data[i][j];
            }
        }
        return *this;
    }

    /// Kronecker product 克罗内克积（张量积）
    template<int r, int c>
    Matrix<DataType, ROW * r, COL * c> kronecker(const Matrix<DataType, r, c>& m) const
    {
        Matrix<DataType, ROW * r, COL * c> ret;
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                for (int a = 0; a < r; ++a)
                {
                    for (int b = 0; b < c; ++b)
                    {
                        ret.data[i * r + a][j * c + b] = data[i][j] * m.data[a][b];
                    }
                }
            }
        }
        return ret;
    }

    /// min到max的随机矩阵
    void random(DataType min, DataType max)
    {
        DataType len = (max - min) / (DataType)RAND_MAX;
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] = min + (DataType)rand() * len;
            }
        }
    }

    /// 调整权值，算法包括隐含转置的单维张量积
    Matrix& adjustW(const Matrix<DataType, 1, ROW>& mX, const Matrix<DataType, 1, COL>& mY, double learnrate)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] += learnrate * mX.data[1][i] * mY.data[1][j];
            }
        }
        return *this;
    }
    /// 调整阈值
    Matrix& adjustT(const Matrix& m, double learnrate)
    {
        for (int i = 0; i < ROW; ++i)
        {
            for (int j = 0; j < COL; ++j)
            {
                data[i][j] += learnrate * m.data[i][j];
            }
        }
        return *this;
    }

public:
    int Row() { return ROW; }
    int Col() { return COL; }

public:
    DataType data[ROW][COL];
};

}