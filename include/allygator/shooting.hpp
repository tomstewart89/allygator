///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef allygator_CORE_OPTCTRL_SHOOTING_HPP_
#define allygator_CORE_OPTCTRL_SHOOTING_HPP_

#include <memory>
#include <stdexcept>
#include <vector>

#include "allygator/action.hpp"

namespace allygator
{

/**
 * @brief This class encapsulates a shooting problem
 *
 * A shooting problem encapsulates the initial state \f$\mathbf{x}_{0}\in\mathcal{M}\f$, a set of
 * running action models and a terminal action model for a discretized trajectory into \f$T\f$
 * nodes. It has three main methods - `calc`, `calc_diff` and `rollout`. The first computes the set
 * of next states and cost values per each node \f$k\f$. Instead, `calc_diff` updates the
 * derivatives of all action models. Finally, `rollout` integrates the system dynamics. This class
 * is used to decouple problem formulation and resolution.
 */
class ShootingProblem
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Initialize the shooting problem and allocate its data
     *
     * @param[in] x0              Initial state
     * @param[in] running_models  Running action models (size \f$T\f$)
     * @param[in] terminal_model  Terminal action model
     */
    ShootingProblem(const Eigen::VectorXd& x0,
                    std::vector<std::unique_ptr<RunningAction> > running_models,
                    std::unique_ptr<TerminalAction> terminal_model);

    ~ShootingProblem() = default;

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
    void calc(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);

    /**
     * @brief Compute the derivatives of the cost and dynamics
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
    void calc_diff(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us);

    double calc_cost();

    /**
     * @brief Return the number of running nodes
     */
    std::size_t get_T() const;

    /**
     * @brief Return the initial state
     */
    const Eigen::VectorXd& get_x0() const;

    /**
     * @brief Return the running models
     */
    RunningAction& get_running_model(std::size_t idx);

    /**
     * @brief Return the terminal model
     */
    TerminalAction& get_terminal_model();

    /**
     * @brief Return the dimension of the state tuple
     */
    std::size_t get_nx() const;

    /**
     * @brief Return the dimension of the tangent space of the state manifold
     */
    std::size_t get_ndx() const;

   protected:
    double cost_;                                                  //!< Total cost
    std::size_t T_;                                                //!< number of running nodes
    Eigen::VectorXd x0_;                                           //!< Initial state
    std::unique_ptr<TerminalAction> terminal_model_;               //!< Terminal action model
    std::vector<std::unique_ptr<RunningAction> > running_models_;  //!< Running action model
};

}  // namespace allygator

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
// #include "allygator/core/optctrl/shooting.hxx"

#endif  // allygator_CORE_OPTCTRL_SHOOTING_HPP_
