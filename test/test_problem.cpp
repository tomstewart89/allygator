#include <gtest/gtest.h>

#include "allygator/problem.hpp"

using namespace allygator;

TEST(Problem, ParticleMotion)
{
    Eigen::Vector2d target(1.0, 2.0);
    allygator::ParticleMotionProblem problem(target);

    Eigen::VectorXd x(4);
    Eigen::VectorXd u(2);

    u << 0.0, 0.0;
    x << 0.0, 0.0, 1.0, 2.0;

    std::cout << problem.d_step_d_u(x, u);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}