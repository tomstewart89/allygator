///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "allygator/shooting.hpp"

#include <iostream>
#include <numeric>

namespace allygator
{

ShootingProblem::ShootingProblem(const Eigen::VectorXd &x0,
                                 std::vector<std::unique_ptr<ActionModel>> &&running_models,
                                 std::unique_ptr<ActionModel> &&terminal_model)
    : cost_(double(0.)), T_(running_models.size()), x0_(x0)
//   terminal_model_(terminal_model),
//   running_models_(running_models)
{
}

void ShootingProblem::calc(const std::vector<Eigen::VectorXd> &xs,
                           const std::vector<Eigen::VectorXd> &us)
{
    for (std::size_t i = 0; i < T_; ++i)
    {
        running_models_[i]->calc(xs[i], us[i]);
    }

    // terminal_model_->calc(xs.back());
}

void ShootingProblem::calcDiff(const std::vector<Eigen::VectorXd> &xs,
                               const std::vector<Eigen::VectorXd> &us)
{
    for (std::size_t i = 0; i < T_; ++i)
    {
        running_models_[i]->calcDiff(xs[i], us[i]);
    }

    // terminal_model_->calcDiff(xs.back());
}

double ShootingProblem::calcCost()
{
    return std::accumulate(
        running_models_.begin(), running_models_.end(), terminal_model_->get_data().cost_,
        [](double sum, const auto &model) { return sum + model->get_data().cost_; });
}

std::size_t ShootingProblem::get_T() const { return T_; }

const Eigen::VectorXd &ShootingProblem::get_x0() const { return x0_; }

ActionModel &ShootingProblem::get_running_model(std::size_t idx) { return *running_models_[idx]; }

ActionModel &ShootingProblem::get_terminal_model() { return *terminal_model_; }

std::size_t ShootingProblem::get_nx() const { return running_models_[0]->get_nx(); }

std::size_t ShootingProblem::get_ndx() const { return running_models_[0]->get_ndx(); }

}  // namespace allygator
