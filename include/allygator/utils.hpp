#pragma once

#include <functional>

#include "Eigen/Core"

namespace allygator
{
inline bool is_oob(const double value) { return (std::isnan(value) || value >= 1e30); }

inline std::vector<std::vector<double>> split(const std::vector<Eigen::VectorXd>& vec)
{
    std::vector<std::vector<double>> out;

    if (!vec.empty())
    {
        for (std::size_t i = 0; i < vec[0].rows(); ++i)
        {
            std::vector<double> row(vec.size());
            std::transform(vec.begin(), vec.end(), row.begin(), [&i](auto& x) { return x[i]; });
            out.push_back(row);
        }
    }

    return out;
}

}  // namespace allygator
