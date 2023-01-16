#pragma once

#include <functional>

#include "Eigen/Core"

namespace allygator
{

Eigen::MatrixXd differentiate(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f,
                              const Eigen::VectorXd& x, double epsilon = 1e-5);

Eigen::MatrixXd differentiate(const std::function<double(const Eigen::VectorXd&)> f,
                              const Eigen::VectorXd& x, double epsilon = 1e-5);

}  // namespace allygator
