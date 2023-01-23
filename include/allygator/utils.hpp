#pragma once

#include <functional>

#include "Eigen/Core"

namespace allygator
{
inline bool is_oob(const double value) { return (std::isnan(value) || value >= 1e30); }

}  // namespace allygator
