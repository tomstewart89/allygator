#include "allygator/numeric_diff.hpp"

namespace allygator
{

Eigen::MatrixXd differentiate(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f,
                              const Eigen::VectorXd& x, double epsilon)
{
    Eigen::VectorXd y = f(x);
    Eigen::MatrixXd J(x.rows(), y.rows());
    Eigen::VectorXd perturbation = Eigen::VectorXd::Zero(x.rows());

    for (int i = 0; i < y.rows(); ++i)
    {
        for (int j = 0; j < x.rows(); ++j)
        {
            perturbation(j) = epsilon;

            J(j, i) = (f(x + perturbation)(i) - y(i)) / epsilon;

            perturbation(j) = 0;
        }
    }

    return J;
}

Eigen::MatrixXd differentiate(const std::function<double(const Eigen::VectorXd&)> f,
                              const Eigen::VectorXd& x, double epsilon)
{
    return differentiate([&f](const Eigen::VectorXd& x)
                         { return (Eigen::VectorXd(1) << f(x)).finished(); },
                         x, epsilon);
}

}  // namespace allygator
