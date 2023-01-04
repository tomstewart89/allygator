#include <gtest/gtest.h>

#include "allygator/numeric_diff.hpp"

using namespace allygator;

TEST(NumericDifferentiation, Univariate)
{
    auto p = (Eigen::VectorXd(1) << 1.0).finished();

    auto f = [](const Eigen::MatrixXd& x)
    { return (Eigen::VectorXd(1) << std::exp(x(0))).finished(); };
    auto f_dot = differentiate(f, p);

    EXPECT_NEAR(f_dot(0), f(p)(0), 1e-3);

    auto g = [](const Eigen::MatrixXd& x)
    { return (Eigen::VectorXd(1) << std::sin(x(0))).finished(); };
    auto g_dot = differentiate(g, p);

    EXPECT_NEAR(g_dot(0), std::cos(p(0)), 1e-3);
}

TEST(NumericDifferentiation, Multivariate)
{
    auto p = (Eigen::VectorXd(3) << 1.0, 2.0, 3.0).finished();

    auto f = [](const Eigen::VectorXd& x)
    { return 5.0 * x(0) * x(0) + 3.5 * x(1) * x(2) + 2.0 * x(0) - x(2); };

    auto f_dot = differentiate(f, p);

    EXPECT_NEAR(f_dot(0), 12.0, 1e-3);
    EXPECT_NEAR(f_dot(1), 10.5, 1e-3);
    EXPECT_NEAR(f_dot(2), 6.0, 1e-3);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}