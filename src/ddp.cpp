#include "allygator/ddp.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>

namespace allygator
{

DDPSolver::DDPSolver(Problem &problem, const Params &params, const std::optional<Callback> cb)
    : problem_(problem), params_(params), cb_(cb)
{
}

std::optional<Trajectory> DDPSolver::solve(Trajectory trajectory)
{
    double reg = params_.reg_init;
    double cost = problem_.calculate_cost(trajectory);
    auto rollout = problem_.do_rollout(trajectory);

    for (std::size_t iter = 0; iter < params_.maxiter; ++iter)
    {
        auto control_law = backward_pass(rollout, reg);

        while (!control_law)
        {
            reg *= params_.reg_incfactor;

            if (reg >= params_.reg_max)
            {
                return std::nullopt;
            }

            control_law = backward_pass(rollout, reg);
        }

        for (const double steplength : generate_step_sizes())
        {
            auto new_trajectory = forward_pass(trajectory, *control_law, steplength);

            if (!new_trajectory)
            {
                continue;
            }

            double new_cost = problem_.calculate_cost(*new_trajectory);

            // Calculate the actual change in cost
            double dV = cost - new_cost;

            // Calculate the expected change in cost
            double dVexp = steplength * (control_law->d[0] + 0.5 * steplength * control_law->d[1]);

            // If the step is in the descent direction of the cost
            if (dVexp >= 0)
            {
                if (control_law->d[0] < params_.th_grad || !rollout.is_feasible ||
                    dV > params_.th_acceptstep * dVexp)
                {
                    trajectory = *new_trajectory;
                    cost = new_cost;

                    // We need to recalculate the derivatives when the step length passes
                    rollout = problem_.do_rollout(trajectory);

                    // If we were only able to take a short step then the quadratic approximation
                    // probably isn't  very accurate so let's increase the regularisation
                    if (steplength > params_.th_stepdec)
                    {
                        reg = std::max(reg / params_.reg_decfactor, params_.reg_min);
                    }
                    // If we were able to take a large step, then we can decrease the regularisation
                    else if (steplength <= params_.th_stepinc)
                    {
                        reg *= params_.reg_incfactor;

                        if (reg >= params_.reg_max)
                        {
                            return std::nullopt;
                        }
                    }

                    break;
                }
            }
        }

        if (rollout.is_feasible && control_law->stop < params_.th_stop)
        {
            return trajectory;
        }
    }

    return std::nullopt;
}

std::optional<ControlLaw> DDPSolver::backward_pass(const Rollout &rollout, const double reg)
{
    const std::size_t T = problem_.get_num_timesteps();
    ControlLaw control_law(T);

    Eigen::MatrixXd Vxx = rollout.Lxx[T];
    Vxx.diagonal().array() += reg;
    Eigen::MatrixXd Vx = rollout.Lx[T] + Vxx * rollout.fs[T];

    for (int t = T - 1; t >= 0; --t)
    {
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> FxTVxx_p = rollout.Fx[t] * Vxx;

        Eigen::MatrixXd Qx = rollout.Lx[t] + rollout.Fx[t].transpose() * Vx;
        Eigen::VectorXd Qu = rollout.Lu[t] + rollout.Fu[t].transpose() * Vx;
        Eigen::MatrixXd Qxx = rollout.Lxx[t] + FxTVxx_p * rollout.Fx[t];
        Eigen::MatrixXd Qxu = rollout.Lxu[t] + FxTVxx_p * rollout.Fu[t];
        Eigen::MatrixXd Quu = rollout.Luu[t] + rollout.Fu[t].transpose() * Vxx * rollout.Fu[t];

        Quu.diagonal().array() += reg;

        Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);

        if (Quu_llt.info() != Eigen::Success)
        {
            return std::nullopt;
        }

        control_law.k[t] = Quu_llt.solve(Qu);
        control_law.K[t] = Quu_llt.solve(Qxu.transpose());

        Vx = Qx - control_law.K[t].transpose() * Qu + Vxx * rollout.fs[t];
        Vxx = Qxx - Qxu * control_law.K[t];
        Vxx.diagonal().array() += reg;

        // Ensure symmetry of Vxx
        Eigen::MatrixXd Vxx_tmp = 0.5 * (Vxx + Vxx.transpose());
        Vxx = Vxx_tmp;

        control_law.stop += Qu.squaredNorm();

        // don't know what this is
        control_law.d[0] += Qu.dot(control_law.k[t]);

        // this is the change in the value at time t
        control_law.d[1] -= control_law.k[t].dot(Quu * control_law.k[t]);

        if (is_oob(Vx.lpNorm<Eigen::Infinity>()) || is_oob(Vxx.lpNorm<Eigen::Infinity>()))
        {
            return std::nullopt;
        }
    }

    return control_law;
}

/**
 * @brief
 *
 * @param steplength initially 1 but will be set to progressively more conservative values... until
 * something(?) happens
 */
std::optional<Trajectory> DDPSolver::forward_pass(const Trajectory &trajectory,
                                                  const ControlLaw &control_law,
                                                  const double steplength)
{
    Trajectory new_traj(trajectory.T());
    new_traj.x[0] = trajectory.x[0];

    for (std::size_t t = 0; t < problem_.get_num_timesteps(); ++t)
    {
        auto dx = trajectory.x[t] - new_traj.x[t];
        new_traj.u[t] = trajectory.u[t] - control_law.k[t] * steplength - control_law.K[t] * dx;

        new_traj.x[t + 1] = problem_.step(new_traj.x[t], new_traj.u[t]);

        if (is_oob(new_traj.x[t + 1].lpNorm<Eigen::Infinity>()))
        {
            return std::nullopt;
        }
    }

    return new_traj;
}

}  // namespace allygator
