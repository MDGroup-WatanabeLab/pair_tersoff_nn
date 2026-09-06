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
   direct C++ Tersoff-NN backend.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(tersoff_nn_direct,PairTersoffNNDirect);
// clang-format on
#else

#ifndef LMP_PAIR_TERSOFF_NN_DIRECT_H
#define LMP_PAIR_TERSOFF_NN_DIRECT_H

#include "pair.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class PairTersoffNNDirect : public Pair {
 public:
  PairTersoffNNDirect(class LAMMPS *);
  ~PairTersoffNNDirect() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  void load_model(const std::string&);
  void load_csv(const std::string&);
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
  bool use_gpu_ = false;

  enum class CutoffFunction {
    TERSOFF,
    SMOOTH
  };

  char* filename = nullptr;
  int read_once_flag = 0;

  int shift_flag;    // flag to turn on/off shift
  double shift;      // negative change in equilibrium bond length

  int max_triplet;
  int max_pair;

  // model load
  std::string line_csv, feature, min_str, max_str;
  double min_val, max_val;
  std::unordered_map<std::string, std::pair<double, double>> stats_map;

  enum class Activation {
    LINEAR,
    SILU,
    SOFTPLUS,
    SHIFTED_SOFTPLUS,
  };

  struct DenseLayer {
    int in = 0;
    int out = 0;
    bool has_bias = false;
    std::vector<double> weight;
    std::vector<double> bias;
  };
  // workspace for MLP backpropagation, reused for each batch of triplets
  struct MLPBatchWorkspace {
    std::vector<int> dims;
    std::vector<std::vector<double>> values;
    std::vector<std::vector<double>> act_derivs;
    std::vector<double> delta;
    std::vector<double> prev_delta;
  };

  struct DirectedPairCache {
    int i = 0;
    int j = 0;
    int itype = 0;
    int jtype = 0;
    int iparam_ij = 0;
    int nn_itype = 0;
    int nn_jtype = 0;
    int triplet_begin = 0;
    int triplet_end = 0;
    double delrij[3] = {0.0, 0.0, 0.0};
    double rij_hat[3] = {0.0, 0.0, 0.0};
    double rijsq = 0.0;
    double rij_inv = 0.0;
    double attractive_scale = 0.0;
    double zeta_nn = 0.0;
    double bij = 1.0;
    double db_dzeta = 0.0;
  };

  struct TripletCache {
    int pair_index = 0;
    int k = 0;
    double fcik = 0.0;
    double dfcik = 0.0;
    double dzeta_draw[3] = {0.0, 0.0, 0.0};
  };

  struct DirectModel {
    int num_types = 0;
    int dim_embed_per_atom = 0;
    double rmin = 0.0;
    double rmax = 0.0;
    CutoffFunction pair_cutoff_function = CutoffFunction::TERSOFF;
    CutoffFunction nn_cutoff_function = CutoffFunction::SMOOTH;
    std::string pair_cutoff_function_name = "tersoff";
    std::string nn_cutoff_function_name = "smooth";
    std::vector<double> bigr_table;
    std::vector<double> bigd_table;
    std::vector<double> p_table;
    std::vector<double> embedding;
    std::vector<DenseLayer> xi_layers;
    std::vector<Activation> xi_activations;
    std::vector<DenseLayer> bij_layers;
    std::vector<Activation> bij_activations;
  };

  DirectModel direct_model;
  bool direct_model_loaded = false;

  // Maximum number of valid triplets evaluated in one direct NN call.
  // This is the inference batch size, not the size of the pair/triplet caches.
  static constexpr int XI_BATCH_CHUNK = 4096;
  // Maximum number of directed pairs evaluated in one BijNN call.
  static constexpr int BIJ_BATCH_CHUNK = 4096;

  // Packed caches for all valid pairs and triplets in the current timestep.
  // clear() resets their logical size while retaining capacity for the next step.
  std::vector<DirectedPairCache> nn_pairs_;
  std::vector<TripletCache> nn_triplets_;

  // Fixed-size inference buffers allocated in init_style() and reused for each
  // group of up to XI_BATCH_CHUNK triplets.
  std::vector<double> xi_batch_inputs_;
  std::vector<double> xi_values_;
  std::vector<double> xi_grads_;
  std::vector<int> xi_batch_triplets_;
  MLPBatchWorkspace xi_mlp_workspace_;

  // Fixed-size inference buffers for groups of up to BIJ_BATCH_CHUNK pairs.
  std::vector<double> bij_batch_inputs_;
  std::vector<double> bij_values_;
  std::vector<double> bij_grads_;
  std::vector<int> bij_batch_pairs_;
  MLPBatchWorkspace bij_mlp_workspace_;
  std::vector<int> nn_type_i_;
  std::vector<int> nn_type_j_;
  std::vector<int> nn_type_k_;
  std::vector<double> embedding_triplet_tails_;

  // bond order
  double b_ij, ex_force_ik, ex_force_jk;
  std::array<double, 6> scaling_factors;
  std::array<double, 6> feature_mins;
  std::array<double, 6> feature_maxs;

  void eval_xi_batch_with_grad(int);
  void eval_bij_batch_with_grad(int);
  void eval_mlp_batch_with_grad(const std::vector<DenseLayer>&,
                                const std::vector<Activation>&, const double*, int, int,
                                double*, double*, MLPBatchWorkspace&) const;
  double fc_manual(double, int, int, int) const;
  double fc_manual_d(double, int, int, int) const;
  int elem_params_manual(int, int, int) const;
  int elem_params_pair_manual(int, int) const;
  int feature_type_to_index(double, int) const;
  void initialize_nn_storage();

  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();
  virtual void repulsive(Param *, double, double &, int, double &);

  void nn_attractive_reg(Param *, double, double, double *, double *, double *, int, double &, double &);
  void nn_attractive_ex(double, double, double, double *, double *, double *, double *, double *);
  virtual double nn_ters_fa(double, Param *);
  virtual double nn_ters_fa_d(double, Param *);

  virtual double ters_fc(double, Param *);
  virtual double ters_fc_d(double, Param *);
};

}    // namespace LAMMPS_NS

#endif
#endif
