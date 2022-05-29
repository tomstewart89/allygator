///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "allygator/ddp.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

bool raiseIfNaN(const double value)
{
    return (std::isnan(value) || std::isinf(value) || value >= 1e30);
}

namespace allygator
{

SolverDDP::SolverDDP(ShootingProblem &problem) : problem_(problem)
{
    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        const std::size_t nu = problem_.get_running_model(t).get_nu();
        const std::size_t nx = problem_.get_running_model(t).get_nx();
        const std::size_t ndx = problem_.get_running_model(t).get_ndx();

        xs_.push_back(Eigen::VectorXd::Zero(nx));
        xs_try_.push_back(Eigen::VectorXd::Zero(nx));

        us_.push_back(Eigen::VectorXd::Zero(nu));
        us_try_.push_back(Eigen::VectorXd::Zero(nu));

        Vxx_.push_back(Eigen::MatrixXd::Zero(ndx, ndx));
        Vx_.push_back(Eigen::VectorXd::Zero(ndx));
        K_.push_back(Eigen::MatrixXd::Zero(nu, ndx));
        k_.push_back(Eigen::VectorXd::Zero(nu));
        fs_.push_back(Eigen::VectorXd::Zero(ndx));
    }

    const std::size_t nx = problem_.get_terminal_model().get_nx();
    const std::size_t ndx = problem_.get_terminal_model().get_ndx();

    xs_try_[0] = problem_.get_x0();
    xs_.push_back(Eigen::VectorXd::Zero(nx));
    xs_try_.push_back(Eigen::VectorXd::Zero(nx));
    Vxx_.push_back(Eigen::MatrixXd::Zero(ndx, ndx));
    Vx_.push_back(Eigen::VectorXd::Zero(ndx));
    fs_.push_back(Eigen::VectorXd::Zero(ndx));

    std::generate_n(std::back_inserter(alphas_), 10,
                    [n = 0]() mutable { return 1.0 / pow(2., static_cast<double>(n++)); });

    alphas_.back() = std::min(alphas_.back(), th_stepinc_);
}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd> &init_xs,
                      const std::vector<Eigen::VectorXd> &init_us, const std::size_t maxiter,
                      const bool is_feasible, const double reginit)
{
    xs_try_[0] = problem_.get_x0();  // it is needed in case that init_xs[0] is infeasible

    std::copy(init_xs.begin(), init_xs.end(), xs_.begin());
    std::copy(init_us.begin(), init_us.end(), us_.begin());
    is_feasible_ = is_feasible;

    reg_ = reginit;
    was_feasible_ = false;

    problem_.calc(xs_, us_);
    calcDiff();

    for (std::size_t iter = 0; iter < maxiter; ++iter)
    {
        while (!backwardPass())
        {
            reg_ *= reg_incfactor_;

            if (reg_ >= reg_max_)
            {
                return false;
            }
        }

        for (const auto &alpha : alphas_)
        {
            steplength_ = alpha;

            if (!forwardPass(steplength_))
            {
                continue;
            }

            // Calculate the actual change in cost
            dV_ = cost_ - cost_try_;

            // Calculate the expected change in cost
            dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

            // If the step is in the descent direction of the cost
            if (dVexp_ >= 0)
            {
                if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_)
                {
                    was_feasible_ = is_feasible_;

                    std::copy(xs_try_.begin(), xs_try_.end(), xs_.begin());
                    std::copy(us_try_.begin(), us_try_.end(), us_.begin());
                    cost_ = cost_try_;

                    // We need to recalculate the derivatives when the step length passes
                    calcDiff();
                    break;
                }
            }
        }

        // If we were only able to take a short step then the quadratic approximation probably isn't
        // very accurate so let's increase the regularisation
        if (steplength_ > th_stepdec_)
        {
            reg_ = std::max(reg_ / reg_decfactor_, reg_min_);
        }
        // If we were able to take a large step, then we can decrease the regularisation
        else if (steplength_ <= th_stepinc_)
        {
            reg_ *= reg_incfactor_;

            if (reg_ >= reg_max_)
            {
                return false;
            }
        }

        std::cout << "it: " << iter << " " << cost_ << " reg: " << reg_ << "\n";

        if (was_feasible_ && stop_ < th_stop_)
        {
            return true;
        }
    }

    return false;
}

double SolverDDP::calcDiff()
{
    problem_.calcDiff(xs_, us_);
    cost_ = problem_.calcCost();

    if (!is_feasible_)
    {
        fs_[0] = problem_.get_running_model(0).get_state().diff(xs_[0], problem_.get_x0());

        for (std::size_t t = 0; t < problem_.get_T(); ++t)
        {
            const auto &model = problem_.get_running_model(t);
            fs_[t + 1] = model.get_state().diff(xs_[t + 1], model.get_data().xnext);
        }

        is_feasible_ = std::all_of(fs_.begin(), fs_.end(),
                                   [this](const Eigen::VectorXd &gap)
                                   { return gap.lpNorm<Eigen::Infinity>() < th_gaptol_; });
    }

    if (!was_feasible_)
    {
        // closing the gaps (because the trajectory is feasible now)
        for (auto &gap : fs_)
        {
            gap.setZero();  // ofcourse this gap must already have an inf-norm of lower than
                            // th_gaptol_ which is crazy small so we probably needn't even do this,
                            // unless this tiny error is being accumulated somewhere
        }
    }
    return cost_;
}

/**
 * @brief
 *
 * @param steplength initially 1 but will be set to progressively more conservative values... until
 * something(?) happens
 */
bool SolverDDP::forwardPass(const double steplength)
{
    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        auto dx = problem_.get_running_model(t).get_state().diff(xs_[t], xs_try_[t]);
        us_try_[t] = us_[t] - k_[t] * steplength - K_[t] * dx;

        problem_.get_running_model(t).calc(xs_try_[t], us_try_[t]);
        xs_try_[t + 1] = problem_.get_running_model(t).get_data().xnext;

        if (raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>()))
        {
            return false;
        }
    }

    // problem_.get_terminal_model().calc(xs_try_.back());

    cost_try_ = problem_.calcCost();

    if (raiseIfNaN(cost_try_))
    {
        return false;
    }

    return true;
}

bool SolverDDP::backwardPass()
{
    d_ = Eigen::Vector2d::Zero();
    stop_ = 0.0;

    Vxx_.back() = problem_.get_terminal_model().get_data().Lxx;
    Vxx_.back().diagonal().array() += reg_;
    Vx_.back() = problem_.get_terminal_model().get_data().Lx + Vxx_.back() * fs_.back();

    for (int t = static_cast<int>(problem_.get_T()) - 1; t >= 0; --t)
    {
        const auto &data = problem_.get_running_model(t).get_data();

        MatrixXdRowMajor FxTVxx_p = data.Fx.transpose() * Vxx_[t + 1];

        Eigen::MatrixXd Qxx = data.Lxx + FxTVxx_p * data.Fx;
        Eigen::MatrixXd Qx = data.Lx + data.Fx.transpose() * Vx_[t + 1];

        Eigen::MatrixXd Qxu = data.Lxu + FxTVxx_p * data.Fu;
        Eigen::MatrixXd Quu = data.Luu + data.Fu.transpose() * Vxx_[t + 1] * data.Fu;
        Eigen::VectorXd Qu = data.Lu + data.Fu.transpose() * Vx_[t + 1];

        Quu.diagonal().array() += reg_;

        Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);

        if (Quu_llt.info() != Eigen::Success)
        {
            std::cout << "not positive definite I guess";
            return false;
        }

        k_[t] = Quu_llt.solve(Qu);
        K_[t] = Quu_llt.solve(Qxu.transpose());

        Vx_[t] = Qx - K_[t].transpose() * Qu + Vxx_[t] * fs_[t];
        Vxx_[t] = Qxx - Qxu * K_[t];

        stop_ += Qu.squaredNorm();
        d_[0] += Qu.dot(k_[t]);           // don't know what this is
        d_[1] -= k_[t].dot(Quu * k_[t]);  // this is the change in the value at time t

        // Ensure symmetry of Vxx
        Eigen::MatrixXd Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
        Vxx_[t] = Vxx_tmp_;
        Vxx_[t].diagonal().array() += reg_;

        if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>()) ||
            raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>()))
        {
            return false;
        }
    }

    return true;
}

}  // namespace allygator
