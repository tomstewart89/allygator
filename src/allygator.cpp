#include <chrono>
#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/gepetto/viewer.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/utils.hpp>
#include <thread>

int main()
{
    const std::string urdf_path = "/home/tom/repos/deep_inverse_dynamics/models/urdf/smpl.urdf";
    pinocchio::ModelTpl<double> model;
    pinocchio::GeometryModel vmodel;

    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::urdf::buildGeom(model, urdf_path, pinocchio::VISUAL, vmodel, pinocchio::rosPaths());

    pinocchio::DataTpl<double> data(model);

    pinocchio::gepetto::Viewer viewer(model, &vmodel, NULL);
    viewer.initViewer("pinocchio");
    viewer.loadViewerModel("humanoid");

    auto q = pinocchio::neutral(model);

    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);

    Eigen::MatrixXd Jc = Eigen::MatrixXd::Zero(6, model.nv);

    for (int i = 0; i < 100; ++i)
    {
        pinocchio::getFrameJacobian(model, data, 3, pinocchio::LOCAL, Jc);

        auto ddq = pinocchio::forwardDynamics(model, data, q, v, Eigen::VectorXd::Zero(model.nv),
                                              Jc, Eigen::VectorXd::Zero(6));

        pinocchio::integrate(model, q, v, q);

        viewer.display(q);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << Jc << "\n";
    }

    // Data* d = static_cast<Data*>(data.get());
    // pinocchio::updateFramePlacement<Scalar>(*state_->get_pinocchio().get(), *d->pinocchio, id_);
    // pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_,
    //                             pinocchio::LOCAL, d->Jc);

    // d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, id_);
    // d->a0 = d->a.toVector();

    // if (gains_[0] != 0.)
    // {
    //     d->rMf = pref_.inverse() * d->pinocchio->oMf[id_];
    //     d->a0 += gains_[0] * pinocchio::log6(d->rMf).toVector();
    // }
    // if (gains_[1] != 0.)
    // {
    //     d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, id_);
    //     d->a0 += gains_[1] * d->v.toVector();
    // }

    // pinocchio::forwardDynamics(pinocchio_, d->pinocchio, d->multibody.actuation->tau,
    //                            d->multibody.contacts->Jc, d->multibody.contacts->a0);

    return 0;
}
