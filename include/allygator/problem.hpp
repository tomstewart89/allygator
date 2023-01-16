#pragma once

#include "numeric_diff.hpp"

using namespace std::placeholders;

namespace allygator
{

struct Trajectory
{
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::VectorXd> u;
};

struct Rollout
{
    std::vector<Eigen::VectorXd> Lu;
    std::vector<Eigen::VectorXd> Lx;
    std::vector<Eigen::MatrixXd> Lxx;
    std::vector<Eigen::MatrixXd> Lxu;
    std::vector<Eigen::MatrixXd> Luu;
    std::vector<Eigen::MatrixXd> Fx;
    std::vector<Eigen::MatrixXd> Fu;
    std::vector<Eigen::VectorXd> xnext;
    std::vector<Eigen::VectorXd> fs;
    double cost = 0.0;
    bool is_feasible;

    Rollout(std::size_t num_timesteps)
        : Lu(num_timesteps),
          Lx(num_timesteps + 1),
          Lxx(num_timesteps + 1),
          Lxu(num_timesteps),
          Luu(num_timesteps),
          Fx(num_timesteps),
          Fu(num_timesteps),
          xnext(num_timesteps + 1),
          fs(num_timesteps + 1)
    {
    }
};

class Problem
{
   public:
    virtual Eigen::VectorXd step(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;

    virtual double running_cost(const unsigned int t, const Eigen::VectorXd& x,
                                const Eigen::VectorXd& u) const = 0;

    virtual double terminal_cost(const Eigen::VectorXd& x) const = 0;

    virtual std::size_t get_num_states() const = 0;

    virtual std::size_t get_num_controls() const = 0;

    virtual std::size_t get_num_timesteps() const = 0;

    virtual Eigen::MatrixXd d_step_dx(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::step, this, _1, u), x);
    }

    virtual Eigen::MatrixXd d_step_du(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::step, this, x, _1), u);
    }

    virtual Eigen::VectorXd d_terminal_cost_dx(const Eigen::VectorXd& x) const
    {
        return differentiate(std::bind(&Problem::terminal_cost, this, _1), x);
    }

    virtual Eigen::MatrixXd dd_terminal_cost_xx(const Eigen::VectorXd& x) const
    {
        return differentiate(std::bind(&Problem::d_terminal_cost_dx, this, _1), x);
    }

    virtual Eigen::VectorXd d_running_cost_dx(const unsigned int t, const Eigen::VectorXd& x,
                                              const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::running_cost, this, t, _1, u), x);
    }

    virtual Eigen::VectorXd d_running_cost_du(const unsigned int t, const Eigen::VectorXd& x,
                                              const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::running_cost, this, t, x, _1), u);
    }

    virtual Eigen::MatrixXd dd_running_cost_xx(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::d_running_cost_dx, this, t, _1, u), x);
    }

    virtual Eigen::MatrixXd dd_running_cost_ux(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::d_running_cost_du, this, t, _1, u), x);
    }

    virtual Eigen::MatrixXd dd_running_cost_uu(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u) const
    {
        return differentiate(std::bind(&Problem::d_running_cost_du, this, t, x, _1), u);
    }

    Trajectory make_trajectory() const
    {
        Trajectory traj;
        traj.x.resize(get_num_timesteps() + 1, Eigen::VectorXd::Zero(get_num_states()));
        traj.u.resize(get_num_timesteps(), Eigen::VectorXd::Zero(get_num_controls()));
        return traj;
    }

    Rollout do_rollout(const Trajectory& trajectory) const
    {
        const auto T = get_num_timesteps();

        Rollout rollout(T);

        rollout.xnext[0] = trajectory.x[0];
        rollout.fs[0] = Eigen::VectorXd::Zero(get_num_states());

        for (std::size_t t = 0; t < T; ++t)
        {
            const auto& x = trajectory.x[t];
            const auto& u = trajectory.u[t];

            rollout.Fx[t] = d_step_dx(x, u);
            rollout.Fu[t] = d_step_du(x, u);
            rollout.Lu[t] = d_running_cost_du(t, x, u);
            rollout.Lx[t] = d_running_cost_dx(t, x, u);
            rollout.Lxx[t] = dd_running_cost_xx(t, x, u);
            rollout.Lxu[t] = dd_running_cost_ux(t, x, u);
            rollout.Luu[t] = dd_running_cost_uu(t, x, u);

            rollout.fs[t + 1] = (trajectory.x[t + 1] - rollout.xnext[t]);
            rollout.xnext[t + 1] = step(x, u);
        }

        rollout.Lx[T] = d_terminal_cost_dx(trajectory.x[T]);
        rollout.Lxx[T] = dd_terminal_cost_xx(trajectory.x[T]);

        rollout.is_feasible = std::all_of(rollout.fs.begin(), rollout.fs.end(),
                                          [this](const Eigen::VectorXd& fs)
                                          { return fs.lpNorm<Eigen::Infinity>() < 1e-16; });

        return rollout;
    }

    double calculate_cost(const Trajectory& trajectory) const
    {
        const auto T = get_num_timesteps();

        double cost = 0.0;
        for (std::size_t t = 0; t < T; ++t)
        {
            cost += running_cost(t, trajectory.x[t], trajectory.u[t]);
        }

        cost += terminal_cost(trajectory.x[T]);

        return cost;
    }
};

class ParticleMotionProblem : public Problem
{
    Eigen::Vector4d target_;
    constexpr static double dt_ = 0.01;
    constexpr static double mass_ = 1.0;

   public:
    ParticleMotionProblem(const Eigen::Vector4d target) : target_(target) {}

    Eigen::VectorXd step(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override
    {
        Eigen::VectorXd out(4);
        out.topRows(2) = x.topRows(2) + x.bottomRows(2) * dt_;
        out.bottomRows(2) = x.bottomRows(2) + u / mass_ * dt_;
        return out;
    }

    double running_cost(const unsigned int t, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u) const override
    {
        return u.squaredNorm();
    }

    double terminal_cost(const Eigen::VectorXd& x) const override
    {
        return (target_ - x).squaredNorm();
    }

    std::size_t get_num_states() const override { return 4; }

    std::size_t get_num_controls() const override { return 2; }

    std::size_t get_num_timesteps() const override { return 50; }
};

}  // namespace allygator
