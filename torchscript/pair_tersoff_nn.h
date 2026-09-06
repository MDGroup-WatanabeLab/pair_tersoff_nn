/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Derived from LAMMPS src/MANYBODY/pair_tersoff.h and modified for the
   TorchScript/LibTorch Tersoff-NN backend.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(tersoff_nn,PairTersoffNN);
// clang-format on
#else

#ifndef LMP_PAIR_TERSOFF_NN_H
#define LMP_PAIR_TERSOFF_NN_H

#include "pair.h"
#include <torch/script.h>

namespace LAMMPS_NS {

class PairTersoffNN : public Pair {
 public:
  PairTersoffNN(class LAMMPS *);
  ~PairTersoffNN() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  void load_model(const std::string&);
  void load_cutoff_function_from_model();
  void load_csv(const std::string&);
  // std::tuple<double, std::vector<std::array<float, 6>>> ters_nn_bij(const torch::Tensor&);
  //void ters_nn_bij(const torch::Tensor&);
  void ters_nn_bij_update(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
  torch::Tensor scale_tensor_minmax(const torch::Tensor&);
  void calc_scaling_factors();

  template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_EITHER> void eval();

  static constexpr int NPARAMS_PER_LINE = 17;

  struct Param {
    double lam1, lam2, lam3;
    double c, d, h;
    double gamma, powerm;
    double powern, beta;
    double biga, bigb, bigd, bigr;
    double cut, cutsq;
    double c1, c2, c3, c4;
    int ielement, jelement, kelement;
    int powermint;
    double Z_i, Z_j;    // added for TersoffZBL
    double ZBLcut, ZBLexpscale;
    double c5, ca1, ca4;    // added for TersoffMOD
    double powern_del;
    double c0;    // added for TersoffMODC
  };

 protected:
  Param *params;      // parameter set for an I-J-K interaction
  double cutmax;      // max cutoff for all elements
  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array
  bool use_gpu_ = false; // flag to use GPU for calculations

  enum class CutoffFunction {
    TERSOFF,
    SMOOTH
  };
  CutoffFunction cutoff_function_ = CutoffFunction::TERSOFF;
  std::string cutoff_function_name_ = "tersoff";

  char* filename = nullptr;
  int read_once_flag = 0;

  int shift_flag;    // flag to turn on/off shift
  double shift;      // negative change in equilibrium bond length

  int max_triplet;
  int max_pair;

  // model load
  torch::jit::script::Module tersoff_nn_model;
  std::string line_csv, feature, min_str, max_str;
  double min_val, max_val;
  std::unordered_map<std::string, std::pair<double, double>> stats_map;

  // bond order
  double b_ij, ex_force_ik, ex_force_jk;
  //std::vector<std::vector<std::array<float, 6>>> grad_original;
  torch::Tensor grad_original;
  torch::Tensor scaled_nn_inputs;
  torch::Tensor nn_inputs_tensor;
  //std::array<double, 6> scaling_factors;
  torch::Tensor scaling_factors;
  torch::Tensor t_min;
  torch::Tensor t_max;

  //std::vector<std::array<double, 6>> nn_inputs;
  //std::vector<std::vector<double>> nn_inputs;
  torch::Tensor nn_inputs;
  torch::Tensor scaled_nn_inputs_list;
  torch::Tensor scaled_nn_inputs_list_structure;
  torch::Tensor b_ij_list;
  std::vector<int> triplet_num_per_pair;
  std::vector<std::vector<int>> triplet_num_per_pair_structure;

  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();
  virtual void repulsive(Param *, double, double &, int, double &);
  virtual double zeta(Param *, double, double, double *, double *);
  virtual void force_zeta(Param *, double, double, double &, double &, int, double &);
  void attractive(Param *, double, double, double, double *, double *, double *, double *,
                  double *);

  void nn_attractive_reg(Param *, double, double, double, double, double *, double *, double *, double *, double *, double *, int, double &, double &);
  void nn_attractive_ex(Param *, double, double, double, double, double, double *, double *, double *, double *, double *, double *);
  virtual double nn_ters_fa(double, Param *);
  virtual double nn_ters_fa_d(double, Param *);

  virtual double ters_fc(double, Param *);
  virtual double ters_fc_d(double, Param *);
  virtual double ters_fa(double, Param *);
  virtual double ters_fa_d(double, Param *);
  virtual double ters_bij(double, Param *);
  virtual double ters_bij_d(double, Param *);

  virtual void ters_zetaterm_d(double, double *, double, double, double *, double, double, double *,
                               double *, double *, Param *);
  void costheta_d(double *, double, double *, double, double *, double *, double *);

  // inlined functions for efficiency

  inline double ters_gijk(const double costheta, const Param *const param) const
  {
    const double ters_c = param->c * param->c;
    const double ters_d = param->d * param->d;
    const double hcth = param->h - costheta;

    return param->gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
  }

  inline double ters_gijk_d(const double costheta, const Param *const param) const
  {
    const double ters_c = param->c * param->c;
    const double ters_d = param->d * param->d;
    const double hcth = param->h - costheta;
    const double numerator = -2.0 * ters_c * hcth;
    const double denominator = 1.0 / (ters_d + hcth * hcth);
    return param->gamma * numerator * denominator * denominator;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
