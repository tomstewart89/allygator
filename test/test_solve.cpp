#include <gtest/gtest.h>

#include "allygator/ddp.hpp"

using namespace allygator;

TEST(Solve, ParticleMotion)
{
    allygator::ParticleMotionProblem problem(Eigen::Vector4d(0, 0, 1.0, 2.0));

    allygator::DDPSolver solver(problem);

    auto traj = solver.solve(problem.make_trajectory());
}

int main(int argc, char** argv)
{
    allygator::ParticleMotionProblem problem(Eigen::Vector4d(0, 0, 1.0, 2.0));

    allygator::DDPSolver solver(problem);

    auto traj = solver.solve(problem.make_trajectory());

    return 0;
    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
}
