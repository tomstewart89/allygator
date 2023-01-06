#pragma once

#include "numeric_diff.hpp"

using namespace std::placeholders;

namespace allygator
{

struct StateAction
{
    Eigen::VectorXd x;
    Eigen::VectorXd u;
};

class Problem
{
   public:
    virtual Eigen::VectorXd step(const Eigen::VectorXd& x, const Eigen::VectorXd& u) = 0;

    virtual double running_cost(const unsigned int t, const Eigen::VectorXd& x,
                                const Eigen::VectorXd& u) = 0;

    virtual double terminal_cost(const Eigen::VectorXd& x) = 0;

    virtual std::size_t get_num_states() = 0;

    virtual std::size_t get_num_controls() = 0;

    virtual std::size_t get_num_timesteps() = 0;

    virtual Eigen::VectorXd get_initial_state() = 0;

    virtual Eigen::MatrixXd d_step_d_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::step, this, _1, u), x);
    }

    virtual Eigen::MatrixXd d_step_d_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::step, this, x, _1), u);
    }

    virtual Eigen::VectorXd d_terminal_cost_x(const Eigen::VectorXd& x)
    {
        return differentiate(std::bind(&Problem::terminal_cost, this, _1), x);
    }

    virtual Eigen::MatrixXd dd_terminal_cost_xx(const Eigen::VectorXd& x)
    {
        return differentiate(std::bind(&Problem::d_terminal_cost_x, this, _1), x);
    }

    virtual Eigen::VectorXd d_running_cost_x(const unsigned int t, const Eigen::VectorXd& x,
                                             const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::running_cost, this, t, _1, u), x);
    }
    virtual Eigen::VectorXd d_running_cost_u(const unsigned int t, const Eigen::VectorXd& x,
                                             const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::running_cost, this, t, x, _1), u);
    }

    virtual Eigen::MatrixXd dd_running_cost_xx(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::d_running_cost_x, this, t, _1, u), x);
    }
    virtual Eigen::MatrixXd dd_running_cost_ux(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::d_running_cost_u, this, t, _1, u), x);
    }
    virtual Eigen::MatrixXd dd_running_cost_uu(const unsigned int t, const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& u)
    {
        return differentiate(std::bind(&Problem::d_running_cost_u, this, t, x, _1), u);
    }

    std::vector<StateAction> make_zero_trajectory()
    {
        std::vector<StateAction> out;
        std::generate_n(std::back_inserter(out), get_num_timesteps() + 1,
                        [this]()
                        {
                            return StateAction{Eigen::VectorXd::Zero(get_num_states()),
                                               Eigen::VectorXd::Zero(get_num_controls())};
                        });

        return out;
    }
};

class ParticleMotionProblem : public Problem
{
    Eigen::Vector4d target_;
    constexpr static double dt_ = 0.01;
    constexpr static double mass_ = 1.0;

   public:
    ParticleMotionProblem(const Eigen::Vector4d target) : target_(target) {}

    Eigen::VectorXd step(const Eigen::VectorXd& x, const Eigen::VectorXd& u) override
    {
        Eigen::VectorXd out(4);
        out.topRows(2) = x.topRows(2) + x.bottomRows(2) * dt_;
        out.bottomRows(2) = x.bottomRows(2) + u / mass_ * dt_;
        return out;
    }

    double running_cost(const unsigned int t, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u) override
    {
        return u.squaredNorm();
    }

    double terminal_cost(const Eigen::VectorXd& x) override { return (target_ - x).squaredNorm(); }

    std::size_t get_num_states() override { return 4; }

    std::size_t get_num_controls() override { return 2; }

    std::size_t get_num_timesteps() override { return 50; }

    Eigen::VectorXd get_initial_state() override { return Eigen::VectorXd::Zero(4); }
};

}  // namespace allygator
