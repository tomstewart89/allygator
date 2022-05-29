///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

namespace allygator
{

struct ActionData
{
    Eigen::MatrixXd Lxx, Lxu, Luu;
    Eigen::VectorXd Lu, Lx, Fx, Fu;
    Eigen::VectorXd xnext;
    double cost_;
};

struct State
{
    virtual Eigen::VectorXd diff(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1) const = 0;
};

/**
 * @brief This class encapsulates a shooting problem
 *
 * A shooting problem encapsulates the initial state \f$\mathbf{x}_{0}\in\mathcal{M}\f$, a set of
 * running action models and a terminal action model for a discretized trajectory into \f$T\f$
 * nodes. It has three main methods - `calc`, `calcDiff` and `rollout`. The first computes the set
 * of next states and cost values per each node \f$k\f$. Instead, `calcDiff` updates the derivatives
 * of all action models. Finally, `rollout` integrates the system dynamics. This class is used to
 * decouple problem formulation and resolution.
 */
class ActionModel
{
   public:
    /**
     * @brief Compute the cost and the next states
     *
     * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control
     * \f$\mathbf{u_{s}}\f$ trajectory, it computes the next state \f$\mathbf{x}_{k+1}\f$ and cost
     * \f$l_{k}\f$.
     *
     * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size \f$T+1\f$)
     * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
     */
    virtual void calc(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) = 0;

    /*
     *
     * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control
     * \f$\mathbf{u_{s}}\f$ trajectory, it computes the derivatives of the cost
     * \f$(\mathbf{l}_{\mathbf{x}}, \mathbf{l}_{\mathbf{u}}, \mathbf{l}_{\mathbf{xx}},
     * \mathbf{l}_{\mathbf{xu}}, \mathbf{l}_{\mathbf{uu}})\f$ and dynamics
     * \f$(\mathbf{f}_{\mathbf{x}}, \mathbf{f}_{\mathbf{u}})\f$.
     *
     * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size \f$T+1\f$)
     * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
     */
    virtual void calcDiff(const Eigen::VectorXd& xs, const Eigen::VectorXd& us) = 0;

    virtual const ActionData& get_data() const = 0;

    virtual const State& get_state() const = 0;

    virtual const std::size_t get_nu() const = 0;

    virtual const std::size_t get_nx() const = 0;

    virtual const std::size_t get_ndx() const = 0;
};

}  // namespace allygator
