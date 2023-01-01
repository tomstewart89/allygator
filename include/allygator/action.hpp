///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Core>
#include <cassert>
#include <vector>

namespace allygator
{

struct State
{
    virtual Eigen::VectorXd diff(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1) const = 0;
};

class Action
{
   public:
    virtual void calc(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) = 0;

    virtual void calc_diff(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) = 0;

    virtual const State& get_state() const = 0;

    virtual const std::size_t get_nu() const = 0;

    virtual const std::size_t get_nx() const = 0;

    virtual const std::size_t get_ndx() const = 0;

    Eigen::MatrixXd Lxx, Lxu, Luu;
    Eigen::VectorXd Lu, Lx, Fx, Fu;
    Eigen::VectorXd xnext;
    double cost_;
};

class RunningAction : public Action
{
};

class TerminalAction : public Action
{
    TerminalAction()
        : Lu(Eigen::VectorXd::Zeros(0)),
          Lxu(Eigen::MatrixXd::Zeros(get_nx(), 0)),
          Luu(Eigen::MatrixXd::Zeros(get_nx(), 0))
    {
    }

    virtual void calc(const Eigen::VectorXd& xs) = 0;

    virtual void calc_diff(const Eigen::VectorXd& xs) = 0;

    void calc(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) override
    {
        assert(us.size == 0);
        calc(x);
    }

    void calc_diff(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) override
    {
        assert(us.size == 0);
        calc_diff(x);
    }

    const std::size_t get_nu() const override { return 0; };
};

}  // namespace allygator
