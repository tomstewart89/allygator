#pragma once

#include "numeric_diff.hpp"

using namespace std::placeholders;

namespace allygator
{

class Problem
{
   public:
    virtual Eigen::VectorXd step(const Eigen::VectorXd& x, const Eigen::VectorXd& u) = 0;

    virtual double running_cost(const unsigned int t, const Eigen::VectorXd& x,
                                const Eigen::VectorXd& u) = 0;

    virtual double terminal_cost(const Eigen::VectorXd& x) = 0;

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
};

class ParticleMotionProblem : public Problem
{
    Eigen::Vector2d target_;
    constexpr static double dt_ = 0.01;
    constexpr static double mass_ = 1.0;

   public:
    ParticleMotionProblem(const Eigen::Vector2d target) : target_(target) {}

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

    double terminal_cost(const Eigen::VectorXd& x) override
    {
        return (target_ - x.topRows(2)).squaredNorm();
    }
};

}  // namespace allygator
