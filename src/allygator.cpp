///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "allygator/ddp.hpp"

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

// #include "allygator/core/costs/cost-sum.hpp"
// #include "allygator/core/costs/residual.hpp"
// #include "allygator/core/integrator/euler.hpp"
// #include "allygator/core/mathbase.hpp"
// #include "allygator/core/residuals/control.hpp"
// #include "allygator/multibody/actions/free-fwddyn.hpp"
// #include "allygator/multibody/actuations/full.hpp"
// #include "allygator/multibody/residuals/frame-placement.hpp"
// #include "allygator/multibody/residuals/state.hpp"
// #include "allygator/multibody/states/multibody.hpp"

// namespace allygator
// {

//     typedef typename allygator::DifferentialActionModelFreeFwdDynamicsTpl<double>
//         DifferentialActionModelFreeFwdDynamics;
//     typedef typename allygator::IntegratedActionModelEulerTpl<double> IntegratedActionModelEuler;
//     typedef typename allygator::ActuationModelFullTpl<double> ActuationModelFull;
//     typedef typename allygator::CostModelSumTpl<double> CostModelSum;
//     typedef typename allygator::CostModelAbstractTpl<double> CostModelAbstract;
//     typedef typename allygator::CostModelResidualTpl<double> CostModelResidual;
//     typedef typename allygator::ResidualModelStateTpl<double> ResidualModelState;
//     typedef typename allygator::ResidualModelFramePlacementTpl<double>
//     ResidualModelFramePlacement; typedef typename allygator::ResidualModelControlTpl<double>
//     ResidualModelControl; typedef typename allygator::MathBaseTpl<double>::Eigen::VectorXd
//     Eigen::VectorXd; typedef typename allygator::MathBaseTpl<double>::Vector3s Vector3s; typedef
//     typename allygator::MathBaseTpl<double>::Matrix3s Matrix3s;
// } // namespace allygator

using namespace allygator;

int main()
{
    // unsigned int N = 100; // number of nodes

    // because urdf is not supported with all double types.
    pinocchio::ModelTpl<double> modeld;
    pinocchio::urdf::buildModel(
        EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", modeld);
    pinocchio::srdf::loadReferenceConfigurations(
        modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf", false);

    // pinocchio::ModelTpl<double> model_full(modeld.cast<double>());
    // pinocchio::ModelTpl<double> model;

    // pinocchio::buildReducedModel(model_full, {5, 6, 7}, Eigen::VectorXd::Zero(model_full.nq),
    // model);

    // auto state = boost::make_shared<allygator::StateMultibodyTpl<double>>(
    //     boost::make_shared<pinocchio::ModelTpl<double>>(model));

    // auto goalTrackingCost = boost::make_shared<CostModelResidual>(
    //     state, boost::make_shared<ResidualModelFramePlacement>(
    //                state, model.getFrameId("gripper_left_joint"),
    //                pinocchio::SE3Tpl<double>(Matrix3s::Identity(),
    //                                          Vector3s(double(0.), double(0), double(.3)))));

    // auto xRegCost =
    //     boost::make_shared<CostModelResidual>(state,
    //     boost::make_shared<ResidualModelState>(state));
    // auto uRegCost = boost::make_shared<CostModelResidual>(
    //     state, boost::make_shared<ResidualModelControl>(state));

    // // Create a cost model per the running and terminal action model.
    // auto runningCostModel = boost::make_shared<CostModelSum>(state);
    // auto terminalCostModel = boost::make_shared<CostModelSum>(state);

    // // Then let's added the running and terminal cost functions
    // runningCostModel->addCost("gripperPose", goalTrackingCost, double(1));
    // runningCostModel->addCost("xReg", xRegCost, double(1e-4));
    // runningCostModel->addCost("uReg", uRegCost, double(1e-4));
    // terminalCostModel->addCost("gripperPose", goalTrackingCost, double(1));

    // // We define an actuation model
    // auto actuation = boost::make_shared<ActuationModelFull>(state);

    // // Next, we need to create an action model for running and terminal knots. The
    // // forward dynamics (computed using ABA) are implemented
    // // inside DifferentialActionModelFullyActuated.
    // auto runningDAM = boost::make_shared<DifferentialActionModelFreeFwdDynamics>(state,
    // actuation,
    //                                                                              runningCostModel);

    // // Building the running and terminal models
    // auto runningModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, double(1e-3));
    // auto terminalModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, double(0.));

    // // Get the initial state
    // auto initial_state =
    //     boost::static_pointer_cast<allygator::StateMultibody>(runningModel->get_state());

    // Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
    // Eigen::VectorXd x0(initial_state->get_nx());
    // x0 << q0, Eigen::VectorXd::Zero(initial_state->get_nv());

    // // For this optimal control problem, we define 100 knots (or running action
    // // models) plus a terminal knot
    // std::vector<boost::shared_ptr<allygator::ActionModelAbstract>> runningModels(N,
    // runningModel); allygator::ShootingProblem problem(x0, runningModels, terminalModel);
    // std::vector<Eigen::VectorXd> xs(N + 1, x0);
    // std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));

    // for (unsigned int i = 0; i < N; ++i)
    // {
    //     const auto &model = problem.get_running_models()[i];
    //     const auto &data = problem.get_runningDatas()[i];
    //     model->quasiStatic(data, us[i], x0);
    // }

    // // Formulating the optimal control problem
    // allygator::SolverDDP ddp(problem);
    // allygator::Timer timer;
    // ddp.solve(xs, us, 100, false, 0.1);

    // std::cout << "NhQ: " << state->get_nq() << std::endl;
    // std::cout << "Number of nodes: " << N << std::endl
    //           << std::endl;
    // std::cout << "cost: " << ddp.get_cost() << "\ntime: " << timer.get_duration() << "\n";

    return 0;
}
