#include <gtest/gtest.h>
#include <matplot/matplot.h>

#include "allygator/problem.hpp"
#include "allygator/utils.hpp"

using namespace allygator;

TEST(Problem, ParticleMotion)
{
    allygator::ParticleMotionProblem problem({1.0, 2.0, 0.0, 0.0});

    auto traj = problem.make_trajectory();
    std::fill(traj.u.begin(), traj.u.end(), Eigen::Vector2d(0.5, 0.25));

    auto sim_traj = problem.simulate(traj);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}