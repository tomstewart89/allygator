///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Cholesky>
#include <optional>
#include <vector>

#include "allygator/problem.hpp"
#include "allygator/utils.hpp"

namespace allygator
{
struct ControlLaw
{
    ControlLaw(const std::size_t num_timesteps) : K(num_timesteps), k(num_timesteps) {}

    std::vector<Eigen::MatrixXd> K;
    std::vector<Eigen::VectorXd> k;

    Eigen::Vector2d d = Eigen::Vector2d::Zero();
    double stop = 0.0;
};

struct SolverState
{
    Trajectory &trajectory;
    Rollout &rollout;
    ControlLaw &control_law;
    double cost;
    double reg;
    std::size_t iter;
};

using Callback = std::function<void(const SolverState &)>;

void log_to_stdout(const SolverState &);

/**
 * @brief Differential Dynamic Programming (DDP) solver
 *
 * The DDP solver computes an optimal trajectory and control commands by iterates running
 * `backward_pass()` and `forward_pass()`. The backward-pass updates locally the quadratic
 * approximation of the problem and computes descent direction. If the warm-start is feasible, then
 * it computes the gaps \f$\mathbf{\bar{f}}_s\f$ and run a modified Riccati sweep: \f{eqnarray*}
 *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
 * (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
 * (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
 * V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{x}_k},\\
 *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
 * V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k},\\
 *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
 * V_{\mathbf{xx}_{k+1}} \mathbf{f}_{\mathbf{u}_k}. \f} Then, the forward-pass rollouts this new
 * policy by integrating the system dynamics along a tuple of optimized control commands
 * \f$\mathbf{u}^*_s\f$, i.e. \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k). \f}
 *
 * \sa `backward_pass()` and `forward_pass()`
 */

struct Params
{
    std::size_t maxiter = 100;
    double reg_init = 1e-9;
    double reg_incfactor = 10.0;  //!< Regularization factor used to increase the damping value
    double reg_decfactor = 10.0;  //!< Regularization factor used to decrease the damping value
    double reg_min = 1e-9;        //!< Minimum allowed regularization value
    double reg_max = 1e9;         //!< Maximum allowed regularization value
    double th_grad = 1e-12;       //!< Tolerance of the expected gradient used for testing the step
    double th_gaptol = 1e-16;     //!< Threshold limit to check non-zero gaps
    double th_stepdec = 0.5;      //!< Step-length threshold used to decrease regularization
    double th_stepinc = 0.01;     //!< Step-length threshold used to increase regularization
    double th_acceptstep = 0.1;   //!< Threshold used for accepting step
    double th_stop = 1e-9;        //!< Tolerance for stopping the algorithm
};

class DDPSolver
{
   public:
    explicit DDPSolver(Problem &problem, const Params &params,
                       const Callback cb = std::bind(log_to_stdout, _1));

    ~DDPSolver() = default;

    std::optional<Trajectory> solve(Trajectory trajectory);

   private:
    /**
     * @brief Run the backward pass (Riccati sweep)
     *
     * It assumes that the Jacobian and Hessians of the optimal control problem have been compute
     * (i.e. `calc_diff()`). The backward pass handles infeasible guess through a modified Riccati
     * sweep: \f{eqnarray*} \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} +
     * \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}}
     * +
     * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
     *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
     * (V_{\mathbf{x}_{k+1}}
     * +
     * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
     *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
     * V_{\mathbf{xx}_{k+1}}
     * \mathbf{f}_{\mathbf{x}_k},\\
     *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
     * V_{\mathbf{xx}_{k+1}}
     * \mathbf{f}_{\mathbf{u}_k},\\
     *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
     * V_{\mathbf{xx}_{k+1}} \mathbf{f}_{\mathbf{u}_k}, \f} where
     * \f$\mathbf{l}_{\mathbf{x}_k}\f$,\f$\mathbf{l}_{\mathbf{u}_k}\f$,\f$\mathbf{f}_{\mathbf{x}_k}\f$
     * and \f$\mathbf{f}_{\mathbf{u}_k}\f$ are the Jacobians of the cost function and dynamics,
     * \f$\mathbf{l}_{\mathbf{xx}_k}\f$,\f$\mathbf{l}_{\mathbf{xu}_k}\f$ and
     * \f$\mathbf{l}_{\mathbf{uu}_k}\f$ are the Hessians of the cost function,
     * \f$V_{\mathbf{x}_{k+1}}\f$ and \f$V_{\mathbf{xx}_{k+1}}\f$ defines the linear-quadratic
     * approximation of the Value function, and \f$\mathbf{\bar{f}}_{k+1}\f$ describes the gaps of
     * the dynamics.
     */
    std::optional<ControlLaw> backward_pass(const Rollout &rollout, const double reg);

    /**
     * @brief Run the forward pass or rollout
     *
     * It rollouts the action model given the computed policy (feedforward terns and feedback gains)
     * by the `backward_pass()`: \f{eqnarray}
     *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
     *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
     * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
     * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k). \f} We can define different step lengths
     * \f$\alpha\f$.
     *
     * @param  stepLength  applied step length (\f$0\leq\alpha\leq1\f$)
     */
    std::optional<Trajectory> forward_pass(const Trajectory &trajectory,
                                           const ControlLaw &control_law, const double steplength);

    std::vector<double> generate_step_sizes() const
    {
        std::vector<double> alphas;

        std::generate_n(std::back_inserter(alphas), 10,
                        [n = 0]() mutable { return 1.0 / pow(2., static_cast<double>(n++)); });

        alphas.back() = std::min(alphas.back(), params_.th_stepinc);

        return alphas;
    }

    Problem &problem_;  //!< optimal control problem
    const Params params_;
    const Callback cb_;
};

}  // namespace allygator
