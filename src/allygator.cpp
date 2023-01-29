#include <matplot/matplot.h>

#include <iostream>

#include "allygator/ddp.hpp"

using namespace allygator;

std::vector<double> generate_timesteps(const std::size_t T)
{
    std::vector<double> t(T);
    std::iota(t.begin(), t.end(), 1.0);
    return t;
}

std::vector<std::string> generate_labels(const std::string& prefix, const std::size_t num)
{
    std::vector<std::string> labels(num);
    std::generate_n(labels.begin(), num,
                    [&prefix, i = 0]() mutable { return prefix + std::to_string(++i); });

    return labels;
}

void log_to_stdout(const DDPSolver::State& state)
{
    using namespace matplot;

    auto t = generate_timesteps(state.trajectory.T() + 1);

    tiledlayout(2, 1);
    auto ax = nexttile();

    hold(on);

    for (const auto& x : split(state.trajectory.x))
    {
        plot(t, x);
    }

    hold(off);

    legend(ax, generate_labels("x", state.trajectory.x[0].rows()));

    ax = nexttile();

    hold(on);

    for (const auto& u : split(state.trajectory.u))
    {
        plot(t, u);
    }

    hold(off);

    legend(ax, generate_labels("u", state.trajectory.u[0].rows()));

    show();
}

int main(int argc, char** argv)
{
    allygator::ParticleMotionProblem problem({200.0, 300.0, 0.0, 0.0});

    allygator::DDPSolver solver(problem, allygator::DDPSolver::Params(),
                                std::bind(log_to_stdout, _1));

    auto traj = solver.solve(problem.make_trajectory());

    return 0;
}
