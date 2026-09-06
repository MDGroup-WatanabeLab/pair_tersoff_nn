// clang-format off
/* ----------------------------------------------------------------------
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
   Contributing author: Aidan Thompson (SNL) - original Tersoff implementation
                        Wengen Ouyang (WHU)  - Shift addition
                        Yusuke Nishimura (Waseda Univ.) - Tersoff-NN direct C++ backend
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Derived from LAMMPS src/MANYBODY/pair_tersoff.cpp and modified for
   Tersoff-NN bond-order evaluation using direct C++ MLP inference.
------------------------------------------------------------------------- */

#include "pair_tersoff_nn_direct.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "math_const.h"
#include "math_extra.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "suffix.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

static constexpr int DELTA = 4;

/* ---------------------------------------------------------------------- */

PairTersoffNNDirect::PairTersoffNNDirect(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  params = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTersoffNNDirect::~PairTersoffNNDirect()
{
  if (copymode) return;

  memory->destroy(params);
  memory->destroy(elem3param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  if (shift_flag) {
    if (evflag) {
      if (eflag) {
        if (vflag_either) eval<1,1,1,1>();
        else eval<1,1,1,0>();
      } else {
        if (vflag_either) eval<1,1,0,1>();
        else eval<1,1,0,0>();
      }
    } else eval<1,0,0,0>();

  } else {

    if (evflag) {
      if (eflag) {
        if (vflag_either) eval<0,1,1,1>();
        else eval<0,1,1,0>();
      } else {
        if (vflag_either) eval<0,1,0,1>();
        else eval<0,1,0,0>();
      }
    } else eval<0,0,0,0>();
  }
}

template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_EITHER>
void PairTersoffNNDirect::eval()
{
  int i,j,k,ii,jj,kk,inum,jnum;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double fforce;
  double rsq,rijsq,riksq,rjksq;
  double delrij[3],delrik[3],delrjk[3],fi[3],fj[3],fk[3];
  double rij_hat[3],rik_hat[3],rjk_hat[3];
  double forceshiftfac;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  const double cutshortsq = cutmax*cutmax;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // obtain max triplets and max pair
  // building neighborlist w/ cutshortsq
  // store as flat neighborlist of all atoms and offsetlist
  // EX) atom0 = [5, 8], atom1 = [2], atom2 = [3, 9, 10]
  // short_buf = [5, 8, 2, 3, 9, 10, ...]
  // short_ofs = [0, 2, 3, 6, ...]
  std::vector<int> short_buf;
  std::vector<int> short_ofs(inum + 1);
  short_ofs[0] = 0;
  max_triplet = 0;
  max_pair = 0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int pair_count_i = 0;
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (SHIFT_FLAG) {
        double rsqtmp = rsq + shift*shift + 2*sqrt(rsq)*shift;
        rsq = rsqtmp;
      }

      if (rsq < cutshortsq) {
        short_buf.push_back(j);
        ++numshort;
        jtype = map[type[j]];
        iparam_ij = elem3param[itype][jtype][jtype];
        if (rsq < params[iparam_ij].cutsq) {
          ++pair_count_i;
        }
      }
    }

    if (pair_count_i > max_pair) {
      max_pair = pair_count_i;
    }

    if (numshort > 0 && (numshort -1) > max_triplet) {
      max_triplet = numshort - 1;
    }
    short_ofs[ii+1] = short_ofs[ii] + numshort;
  }

  // calculate energy, force, stress
  const int xi_input_size = 3 + 3 * direct_model.dim_embed_per_atom;

  // Reuse packed caches from the previous timestep. clear() keeps the
  // allocated memory; push_back() below appends only valid pairs/triplets.
  std::vector<DirectedPairCache> &nn_pairs = nn_pairs_;
  std::vector<TripletCache> &nn_triplets = nn_triplets_;
  nn_pairs.clear();
  nn_triplets.clear();
  if (max_pair > 0) {
    nn_pairs.reserve(static_cast<size_t>(inum) * static_cast<size_t>(max_pair));
  }

  // make prediction batch for xi MLP, size of xi_batch_inputs_ = XI_BATCH_CHUNK * xi_input_size
  // These buffers hold only the next inference batch, not all triplets.
  // After 4096 triplets are evaluated they are overwritten by the next batch.
  // see initialize_nn_storage() for initial memory allocation.
  double *const xi_batch_inputs = xi_batch_inputs_.data();
  double *const xi_values = xi_values_.data();
  double *const xi_grads = xi_grads_.data();
  int *const xi_batch_triplets = xi_batch_triplets_.data();
  int xi_batch_size = 0;

  // References to the type index for each triplet before embedding
  const std::vector<int> &nn_type_i = nn_type_i_;
  const std::vector<int> &nn_type_j = nn_type_j_;
  const std::vector<int> &nn_type_k = nn_type_k_;

  // contains the embedded atom types, size = num_triplets * 3 * dim_embed_per_atom
  const int embedding_tail_size = 3 * direct_model.dim_embed_per_atom;
  const std::vector<double> &embedding_triplet_tails = embedding_triplet_tails_;

  auto flush_xi_batch = [&]() {
    if (xi_batch_size == 0) return;
    eval_xi_batch_with_grad(xi_batch_size);// MLP prediction and backpropagation for the batch of triplets
    for (int local_idx = 0; local_idx < xi_batch_size; ++local_idx) {
      const int trip_idx = xi_batch_triplets[local_idx];
      TripletCache &triplet = nn_triplets[trip_idx]; // reference to the nn_triplets cache
      DirectedPairCache &pair_record = nn_pairs[triplet.pair_index]; // reference to the nn_pairs cache
      const double xi = xi_values[local_idx];
      const double *dxi_dinput = xi_grads + static_cast<size_t>(local_idx) * xi_input_size;
      pair_record.zeta_nn += xi * triplet.fcik;
      triplet.dzeta_draw[0] = dxi_dinput[0] * scaling_factors[0] * triplet.fcik;
      triplet.dzeta_draw[1] = dxi_dinput[1] * scaling_factors[1] * triplet.fcik + xi * triplet.dfcik;
      triplet.dzeta_draw[2] = dxi_dinput[2] * scaling_factors[2] * triplet.fcik;
    }
    xi_batch_size = 0;
  };

  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    const int start = short_ofs[ii];
    const int stop  = short_ofs[ii+1];
    const int numshort = stop - start;

    // 2 body interaction (repulsive)
    for (jj = 0; jj < numshort; jj++){
      j = short_buf[start + jj];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      // shift rsq and store correction for force
      if (SHIFT_FLAG) {
        double rsqtmp = rsq + shift*shift + 2*sqrt(rsq)*shift;
        forceshiftfac = sqrt(rsqtmp/rsq);
        rsq = rsqtmp;
      }

      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];
      if (rsq >= params[iparam_ij].cutsq) continue;

      repulsive(&params[iparam_ij],rsq,fpair,EFLAG,evdwl);

      // correct force for shift in rsq

      if (SHIFT_FLAG) fpair *= forceshiftfac;

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;

      if (EVFLAG) ev_tally(i,j,nlocal,newton_pair,
                          evdwl,0.0,fpair,delx,dely,delz);
    }

    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;

    // 3 body interaction (attractive)
    // Build Xi inputs and evaluate Xi with input gradients in batches
    for (jj = 0; jj < numshort; jj++) {
      j = short_buf[start + jj];
      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];
      // sign is inveerted from repulsive, respect to LAMMPS convention
      delrij[0] = x[j][0] - xtmp;
      delrij[1] = x[j][1] - ytmp;
      delrij[2] = x[j][2] - ztmp;
      const double rijsq_raw = delrij[0]*delrij[0] + delrij[1]*delrij[1] + delrij[2]*delrij[2];
      const double rij_raw = sqrt(rijsq_raw);
      rijsq = rijsq_raw;

      if constexpr (SHIFT_FLAG)
        rijsq += shift*shift + 2*rij_raw*shift;

      if (rijsq >= params[iparam_ij].cutsq) continue;

      const double rij_inv = 1.0/rij_raw;
      scale3(rij_inv, delrij, rij_hat);
      const int nn_itype = nn_type_i[itype];
      const int nn_jtype = nn_type_j[jtype];
      const double scaled_rij = (rij_raw - feature_mins[0]) * scaling_factors[0];
      const int pair_index = static_cast<int>(nn_pairs.size());

      DirectedPairCache pair_record;
      pair_record.i = i;
      pair_record.j = j;
      pair_record.itype = itype;
      pair_record.jtype = jtype;
      pair_record.iparam_ij = iparam_ij;
      pair_record.nn_itype = nn_itype;
      pair_record.nn_jtype = nn_jtype;
      pair_record.triplet_begin = static_cast<int>(nn_triplets.size());
      pair_record.delrij[0] = delrij[0];
      pair_record.delrij[1] = delrij[1];
      pair_record.delrij[2] = delrij[2];
      pair_record.rij_hat[0] = rij_hat[0];
      pair_record.rij_hat[1] = rij_hat[1];
      pair_record.rij_hat[2] = rij_hat[2];
      pair_record.rijsq = rijsq;
      pair_record.rij_inv = rij_inv;
      double rij_attractive;
      if constexpr (SHIFT_FLAG) rij_attractive = sqrt(rijsq);
      else rij_attractive = rij_raw;
      pair_record.attractive_scale =
          0.5 * ters_fc(rij_attractive, &params[iparam_ij]) *
          nn_ters_fa(rij_attractive, &params[iparam_ij]);
      nn_pairs.push_back(pair_record);

      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;

        k = short_buf[start + kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];
        // sign is inverted from repulsive, respect to LAMMPS convention
        delrik[0] = x[k][0] - xtmp;
        delrik[1] = x[k][1] - ytmp;
        delrik[2] = x[k][2] - ztmp;
        const double riksq_raw = delrik[0]*delrik[0] + delrik[1]*delrik[1] + delrik[2]*delrik[2];
        const double rik_raw = sqrt(riksq_raw);
        riksq = riksq_raw;

        if constexpr (SHIFT_FLAG)
          riksq += shift*shift + 2*rik_raw*shift;

        if (riksq >= params[iparam_ijk].cutsq) continue;

        const double rik_inv = 1.0/rik_raw;
        scale3(rik_inv, delrik, rik_hat);

        const int nn_ktype = nn_type_k[ktype];
        const double cosjik = dot3(rij_hat, rik_hat);

        TripletCache triplet_record;
        triplet_record.pair_index = pair_index;
        triplet_record.k = k;
        triplet_record.fcik = fc_manual(rik_raw, nn_itype, nn_jtype, nn_ktype);
        triplet_record.dfcik = fc_manual_d(rik_raw, nn_itype, nn_jtype, nn_ktype);
        nn_triplets.push_back(triplet_record);

        double *xi_input = xi_batch_inputs + static_cast<size_t>(xi_batch_size) * xi_input_size;
        xi_input[0] = scaled_rij;
        xi_input[1] = (rik_raw - feature_mins[1]) * scaling_factors[1];
        xi_input[2] = (cosjik - feature_mins[2]) * scaling_factors[2];

        const double *embedding_tail = embedding_triplet_tails.data() +
            static_cast<size_t>(elem_params_manual(nn_itype, nn_jtype, nn_ktype)) *
            embedding_tail_size;
        std::copy(embedding_tail, embedding_tail + embedding_tail_size, xi_input + 3);
        xi_batch_triplets[xi_batch_size] = static_cast<int>(nn_triplets.size()) - 1;
        ++xi_batch_size;
        if (xi_batch_size == XI_BATCH_CHUNK) flush_xi_batch();
      }

      nn_pairs[pair_index].triplet_end = static_cast<int>(nn_triplets.size());
    }
  }
  flush_xi_batch(); // rest of the triplets in the last batch

  double *const bij_batch_inputs = bij_batch_inputs_.data();
  double *const bij_values = bij_values_.data();
  double *const bij_grads = bij_grads_.data();
  int *const bij_batch_pairs = bij_batch_pairs_.data();
  int bij_batch_size = 0;

  auto flush_bij_batch = [&]() {
    if (bij_batch_size == 0) return;
    eval_bij_batch_with_grad(bij_batch_size);
    for (int local_idx = 0; local_idx < bij_batch_size; ++local_idx) {
      DirectedPairCache &pair_record = nn_pairs[bij_batch_pairs[local_idx]];
      const double bij_nn = bij_values[local_idx];
      const double dbij_nn_dzeta = bij_grads[local_idx];
      const int pair_param = elem_params_pair_manual(pair_record.nn_itype, pair_record.nn_jtype);
      const double p = direct_model.p_table[pair_param];
      pair_record.bij = std::exp(-p * std::log1p(bij_nn));
      pair_record.db_dzeta = -p * pair_record.bij / (1.0 + bij_nn) * dbij_nn_dzeta;
    }
    bij_batch_size = 0;
  };

  for (int pair_index = 0; pair_index < static_cast<int>(nn_pairs.size()); ++pair_index) {
    DirectedPairCache &pair_record = nn_pairs[pair_index];
    if (pair_record.triplet_begin == pair_record.triplet_end) { // two atom case
      pair_record.bij = 1.0;
      pair_record.db_dzeta = 0.0;
    } else {
      bij_batch_inputs[bij_batch_size] = pair_record.zeta_nn;
      bij_batch_pairs[bij_batch_size] = pair_index;
      ++bij_batch_size;
      if (bij_batch_size == BIJ_BATCH_CHUNK) flush_bij_batch();
    }
  }
  flush_bij_batch();

  for (DirectedPairCache &pair_record : nn_pairs) {
    i = pair_record.i;
    j = pair_record.j;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    double fjxtmp = 0.0;
    double fjytmp = 0.0;
    double fjztmp = 0.0;
    double dbijdrij = 0.0;

    for (int trip_idx = pair_record.triplet_begin; trip_idx < pair_record.triplet_end; ++trip_idx) {
      const TripletCache &triplet = nn_triplets[trip_idx];
      k = triplet.k;
      // sign is inverted from repulsive, respect to LAMMPS convention
      delrik[0] = x[k][0] - xtmp;
      delrik[1] = x[k][1] - ytmp;
      delrik[2] = x[k][2] - ztmp;
      const double riksq_raw = delrik[0]*delrik[0] + delrik[1]*delrik[1] + delrik[2]*delrik[2];
      const double rik_raw = sqrt(riksq_raw);
      riksq = riksq_raw;
      if constexpr (SHIFT_FLAG)
        riksq += shift*shift + 2*rik_raw*shift;

      const double rik_inv = 1.0/rik_raw;
      scale3(rik_inv, delrik, rik_hat);
      // sign is inverted from repulsive, respect to LAMMPS convention
      delrjk[0] = x[k][0] - x[j][0];
      delrjk[1] = x[k][1] - x[j][1];
      delrjk[2] = x[k][2] - x[j][2];
      rjksq = delrjk[0]*delrjk[0] + delrjk[1]*delrjk[1] + delrjk[2]*delrjk[2];

      const double rjk = sqrt(rjksq);
      const double rjk_inv = 1.0/rjk;
      scale3(rjk_inv, delrjk, rjk_hat);

      double dbijdrik = 0.0;
      double dbijdrjk = 0.0;

      const double g_rij = pair_record.db_dzeta * triplet.dzeta_draw[0];
      const double g_rik = pair_record.db_dzeta * triplet.dzeta_draw[1];
      const double g_cos = pair_record.db_dzeta * triplet.dzeta_draw[2];

      double rij_for_dcos, rik_for_dcos;
      if constexpr (SHIFT_FLAG) {
        rij_for_dcos = sqrt(pair_record.rijsq);
        rik_for_dcos = sqrt(riksq);
      } else {
        rij_for_dcos = 1.0 / pair_record.rij_inv;
        rik_for_dcos = rik_raw;
      }

      const double dcosjikdrij =
          (pair_record.rijsq - riksq + rjksq) / (2 * pair_record.rijsq * rik_for_dcos);
      const double dcosjikdrik =
          (-pair_record.rijsq + riksq + rjksq) / (2 * riksq * rij_for_dcos);
      const double dcosjikdrjk = -rjk / (rij_for_dcos * rik_for_dcos);

      dbijdrij += g_rij + g_cos * dcosjikdrij;
      dbijdrik += g_rik + g_cos * dcosjikdrik;
      dbijdrjk +=         g_cos * dcosjikdrjk;

      nn_attractive_ex(pair_record.attractive_scale, dbijdrik, dbijdrjk,
                       rik_hat, rjk_hat, fi, fj, fk);

      if (VFLAG_EITHER) v_tally3(i, j, k, fj, fk, pair_record.delrij, delrik);

      f[i][0] += fi[0];
      f[i][1] += fi[1];
      f[i][2] += fi[2];
      fjxtmp += fj[0];
      fjytmp += fj[1];
      fjztmp += fj[2];
      f[k][0] += fk[0];
      f[k][1] += fk[1];
      f[k][2] += fk[2];
    }

    double evdwl_pair = 0.0;
    b_ij = pair_record.bij;
    nn_attractive_reg(&params[pair_record.iparam_ij], dbijdrij, pair_record.rijsq,
                      pair_record.rij_hat, fi, fj, EFLAG, evdwl_pair, fforce);

    fpair = fforce * pair_record.rij_inv;

    if (EVFLAG) ev_tally(i, j, nlocal, newton_pair,
                         evdwl_pair, 0.0, -fpair,
                         -pair_record.delrij[0], -pair_record.delrij[1], -pair_record.delrij[2]);

    f[i][0] += fi[0];
    f[i][1] += fi[1];
    f[i][2] += fi[2];
    f[j][0] += fjxtmp + fj[0];
    f[j][1] += fjytmp + fj[1];
    f[j][2] += fjztmp + fj[2];
  }
  if (vflag_fdotr) virial_fdotr_compute();
  //std::exit(0);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTersoffNNDirect::settings(int narg, char **arg)
{
  // default values

  shift_flag = 0;

  // process optional keywords

  int iarg = 0;
  const char* model_name = nullptr;
  const char* input_stats = nullptr;


  while (iarg < narg) {
    if (strcmp(arg[iarg],"shift") == 0) {
      if (suffix_flag & (Suffix::INTEL|Suffix::GPU|Suffix::KOKKOS))
        error->all(FLERR,"Keyword 'shift' not supported for this style");
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command");
      shift = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      shift_flag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "model_name") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR,"Model name is null");
      }
      model_name = arg[iarg+1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "input_stats") == 0) {
      if (iarg + 1 >= narg) {
        error->all(FLERR,"Input stats file name is null");
      }
      input_stats = arg[iarg+1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "use_gpu") == 0) {
      if(iarg + 1 >= narg) {
        error->all(FLERR,"Missing value after use_gpu keyword");
      }
      const char* val = arg[iarg + 1];
      if (strcmp(val, "yes") == 0 || strcmp(val, "true") == 0 || strcmp(val, "1") == 0){
        error->all(FLERR,"The tersoff_nn_direct pair style does not support GPU execution");
      } else if (strcmp(val, "no") == 0 || strcmp(val, "false") == 0 || strcmp(val, "0") == 0) {
        use_gpu_ = false;
      } else {
        error->all(FLERR,"Illegal use_gpu value (expect yes/no)");
      }
      iarg += 2;
    } else {
      error->all(FLERR,"Illegal pair_style command");
    }
  }

  if (!input_stats) error->all(FLERR,"input_stats is required for tersoff_nn_direct");
  if (!model_name)  error->all(FLERR,"model_name is required for tersoff_nn_direct");
  load_csv(input_stats);
  load_model(model_name);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTersoffNNDirect::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg-3,arg+3);

  // read potential file and initialize potential parameters

  read_file(arg[2]);

  setup_params();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTersoffNNDirect::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Tersoff_nn requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Tersoff_nn requires newton pair on");

  initialize_nn_storage();

  // need a full neighbor list

  neighbor->add_request(this,NeighConst::REQ_FULL);

}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::initialize_nn_storage()
{
  const int xi_input_size = 3 + 3 * direct_model.dim_embed_per_atom;
  const size_t batch_input_size = static_cast<size_t>(XI_BATCH_CHUNK) * xi_input_size;

  xi_batch_inputs_.resize(batch_input_size);
  xi_values_.resize(XI_BATCH_CHUNK);
  xi_grads_.resize(batch_input_size);
  xi_batch_triplets_.resize(XI_BATCH_CHUNK);

  bij_batch_inputs_.resize(BIJ_BATCH_CHUNK);
  bij_values_.resize(BIJ_BATCH_CHUNK);
  bij_grads_.resize(BIJ_BATCH_CHUNK);
  bij_batch_pairs_.resize(BIJ_BATCH_CHUNK);

  MLPBatchWorkspace &workspace = xi_mlp_workspace_;
  const size_t nlayers = direct_model.xi_layers.size();
  workspace.dims.resize(nlayers + 1);
  workspace.values.resize(nlayers + 1);
  workspace.act_derivs.resize(nlayers);
  workspace.values[0].reserve(batch_input_size);

  int max_width = xi_input_size;
  for (size_t ilayer = 0; ilayer < nlayers; ++ilayer) {
    const int width = direct_model.xi_layers[ilayer].out;
    workspace.values[ilayer + 1].reserve(static_cast<size_t>(XI_BATCH_CHUNK) * width);
    workspace.act_derivs[ilayer].reserve(static_cast<size_t>(XI_BATCH_CHUNK) * width);
    max_width = std::max(max_width, width);
  }
  const size_t max_batch_workspace = static_cast<size_t>(XI_BATCH_CHUNK) * max_width;
  workspace.delta.reserve(max_batch_workspace);
  workspace.prev_delta.reserve(max_batch_workspace);

  MLPBatchWorkspace &bij_workspace = bij_mlp_workspace_;
  const size_t bij_nlayers = direct_model.bij_layers.size();
  bij_workspace.dims.resize(bij_nlayers + 1);
  bij_workspace.values.resize(bij_nlayers + 1);
  bij_workspace.act_derivs.resize(bij_nlayers);
  bij_workspace.values[0].reserve(BIJ_BATCH_CHUNK);

  int bij_max_width = 1;
  for (size_t ilayer = 0; ilayer < bij_nlayers; ++ilayer) {
    const int width = direct_model.bij_layers[ilayer].out;
    bij_workspace.values[ilayer + 1].reserve(static_cast<size_t>(BIJ_BATCH_CHUNK) * width);
    bij_workspace.act_derivs[ilayer].reserve(static_cast<size_t>(BIJ_BATCH_CHUNK) * width);
    bij_max_width = std::max(bij_max_width, width);
  }
  const size_t bij_max_batch_workspace =
      static_cast<size_t>(BIJ_BATCH_CHUNK) * bij_max_width;
  bij_workspace.delta.reserve(bij_max_batch_workspace);
  bij_workspace.prev_delta.reserve(bij_max_batch_workspace);

  nn_type_i_.resize(nelements);
  nn_type_j_.resize(nelements);
  nn_type_k_.resize(nelements);
  for (int elem = 0; elem < nelements; ++elem) {
    const double elem_value = static_cast<double>(elem);
    nn_type_i_[elem] = feature_type_to_index(elem_value, 3);
    nn_type_j_[elem] = feature_type_to_index(elem_value, 4);
    nn_type_k_[elem] = feature_type_to_index(elem_value, 5);
  }

  const int embedding_tail_size = 3 * direct_model.dim_embed_per_atom;
  const int ntypes = direct_model.num_types;
  embedding_triplet_tails_.resize(
      static_cast<size_t>(ntypes) * ntypes * ntypes * embedding_tail_size);
  for (int ti = 0; ti < ntypes; ++ti) {
    for (int tj = 0; tj < ntypes; ++tj) {
      for (int tk = 0; tk < ntypes; ++tk) {
        double *tail = embedding_triplet_tails_.data() +
            static_cast<size_t>(elem_params_manual(ti, tj, tk)) * embedding_tail_size;
        const double *emb_i = direct_model.embedding.data() +
            static_cast<size_t>(ti) * direct_model.dim_embed_per_atom;
        const double *emb_j = direct_model.embedding.data() +
            static_cast<size_t>(tj) * direct_model.dim_embed_per_atom;
        const double *emb_k = direct_model.embedding.data() +
            static_cast<size_t>(tk) * direct_model.dim_embed_per_atom;
        std::copy(emb_i, emb_i + direct_model.dim_embed_per_atom, tail);
        std::copy(emb_j, emb_j + direct_model.dim_embed_per_atom,
                  tail + direct_model.dim_embed_per_atom);
        std::copy(emb_k, emb_k + direct_model.dim_embed_per_atom,
                  tail + 2 * direct_model.dim_embed_per_atom);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTersoffNNDirect::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status\n" + Info::get_pair_coeff_status(lmp));

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "tersoff_nn", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement  = ielement;
        params[nparams].jelement  = jelement;
        params[nparams].kelement  = kelement;
        params[nparams].powerm    = values.next_double();
        params[nparams].gamma     = values.next_double();
        params[nparams].lam3      = values.next_double();
        params[nparams].c         = values.next_double();
        params[nparams].d         = values.next_double();
        params[nparams].h         = values.next_double();
        params[nparams].powern    = values.next_double();
        params[nparams].beta      = values.next_double();
        params[nparams].lam2      = values.next_double();
        params[nparams].bigb      = values.next_double();
        params[nparams].bigr      = values.next_double();
        params[nparams].bigd      = values.next_double();
        params[nparams].lam1      = values.next_double();
        params[nparams].biga      = values.next_double();
        params[nparams].powermint = int(params[nparams].powerm);

        if (unit_convert) {
          params[nparams].biga *= conversion_factor;
          params[nparams].bigb *= conversion_factor;
        }
      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      // currently only allow m exponent of 1 or 3
      if (params[nparams].c < 0.0 ||
          params[nparams].d < 0.0 ||
          params[nparams].powern < 0.0 ||
          params[nparams].beta < 0.0 ||
          params[nparams].lam2 < 0.0 ||
          params[nparams].bigb < 0.0 ||
          params[nparams].bigr < 0.0 ||
          params[nparams].bigd < 0.0 ||
          params[nparams].bigd > params[nparams].bigr ||
          params[nparams].lam1 < 0.0 ||
          params[nparams].biga < 0.0 ||
          params[nparams].powerm - params[nparams].powermint != 0.0 ||
          (params[nparams].powermint != 3 &&
           params[nparams].powermint != 1) ||
          params[nparams].gamma < 0.0)
        error->one(FLERR,"Illegal Tersoff_NN parameter");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::setup_params()
{
  int i,j,k,m,n;

  // set elem3param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem3param);
  memory->create(elem3param,nelements,nelements,nelements,"pair:elem3param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has a duplicate entry for: {} {} {}",
                                   elements[i], elements[j], elements[k]);
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry for: {} {} {}",
                              elements[i], elements[j], elements[k]);
        elem3param[i][j][k] = n;
      }


  // compute parameter values derived from inputs

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].bigr + params[m].bigd;
    params[m].cutsq = params[m].cut*params[m].cut;

    if (params[m].powern > 0.0) {
      params[m].c1 = pow(2.0*params[m].powern*1.0e-16,-1.0/params[m].powern);
      params[m].c2 = pow(2.0*params[m].powern*1.0e-8,-1.0/params[m].powern);
      params[m].c3 = 1.0/params[m].c2;
      params[m].c4 = 1.0/params[m].c1;
    } else {
      params[m].c1 = params[m].c2 = params[m].c3 = params[m].c4 = 0.0;
    }
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++)
    if (params[m].cut > cutmax) cutmax = params[m].cut;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::repulsive(Param *param, double rsq, double &fforce,
                            int eflag, double &eng)
{
  double r,tmp_fc,tmp_fc_d,tmp_exp;

  r = sqrt(rsq);
  tmp_fc = ters_fc(r,param);
  tmp_fc_d = ters_fc_d(r,param);
  tmp_exp = exp(-param->lam1 * r);
  fforce = 0.5 * -param->biga * tmp_exp * (tmp_fc_d - tmp_fc*param->lam1) / r;
  if (eflag) eng = 0.5 * tmp_fc * param->biga * tmp_exp;
}

/* ---------------------------------------------------------------------- */

double PairTersoffNNDirect::ters_fc(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 1.0;
  if (r > ters_R+ters_D) return 0.0;

  const double phase = MY_PI2*(r - ters_R)/ters_D;
  if (direct_model.pair_cutoff_function == CutoffFunction::SMOOTH)
    return 0.5 - 0.5625*sin(phase) - 0.0625*sin(3.0*phase);

  return 0.5*(1.0 - sin(phase));
}

/* ---------------------------------------------------------------------- */

double PairTersoffNNDirect::ters_fc_d(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 0.0;
  if (r > ters_R+ters_D) return 0.0;

  const double phase = MY_PI2*(r - ters_R)/ters_D;
  if (direct_model.pair_cutoff_function == CutoffFunction::SMOOTH)
    return -(MY_PI2/ters_D) *
      (0.5625*cos(phase) + 0.1875*cos(3.0*phase));

  return -(MY_PI4/ters_D) * cos(phase);
}

/* ---------------------------------------------------------------------- */

namespace {

void expect_token(std::istream& is, const std::string& expected)
{
  std::string got;
  if (!(is >> got) || got != expected) {
    throw std::runtime_error("Expected token '" + expected + "', got '" + got + "'");
  }
}

std::vector<double> read_doubles(std::istream& is, size_t count)
{
  std::vector<double> values(count);
  for (size_t i = 0; i < count; ++i) {
    if (!(is >> values[i])) throw std::runtime_error("Unexpected EOF while reading doubles");
  }
  return values;
}

} // namespace

void PairTersoffNNDirect::load_model(const std::string& model_path) {
  try {
    std::ifstream in(model_path);
    if (!in) throw std::runtime_error("Cannot open direct NN model file: " + model_path);

    std::string format_version;
    if (!(in >> format_version))
      throw std::runtime_error("Direct NN model file is empty");

    auto parse_cutoff_function = [&](const std::string& name) {
      if (name == "tersoff") return CutoffFunction::TERSOFF;
      if (name == "smooth") return CutoffFunction::SMOOTH;
      throw std::runtime_error("Unsupported cutoff function: " + name);
    };

    if (format_version == "tersoff_nn_direct_v2") {
      expect_token(in, "pair_cutoff_function");
      in >> direct_model.pair_cutoff_function_name;
      direct_model.pair_cutoff_function =
        parse_cutoff_function(direct_model.pair_cutoff_function_name);

      expect_token(in, "nn_cutoff_function");
      in >> direct_model.nn_cutoff_function_name;
      direct_model.nn_cutoff_function =
        parse_cutoff_function(direct_model.nn_cutoff_function_name);
    } else if (format_version == "tersoff_nn_direct_v1") {
      // The v1 exporter was tied to the legacy model: the pair calculation
      // used the Tersoff cutoff while the NN zeta accumulation used smooth.
      direct_model.pair_cutoff_function = CutoffFunction::TERSOFF;
      direct_model.nn_cutoff_function = CutoffFunction::SMOOTH;
      direct_model.pair_cutoff_function_name = "tersoff";
      direct_model.nn_cutoff_function_name = "smooth";
    } else {
      throw std::runtime_error("Unsupported direct NN model format: " + format_version);
    }

    expect_token(in, "num_types");
    in >> direct_model.num_types;
    expect_token(in, "dim_embed_per_atom");
    in >> direct_model.dim_embed_per_atom;
    expect_token(in, "rmin");
    in >> direct_model.rmin;
    expect_token(in, "rmax");
    in >> direct_model.rmax;

    int count = 0;
    expect_token(in, "bigr_table");
    in >> count;
    direct_model.bigr_table = read_doubles(in, count);
    expect_token(in, "bigd_table");
    in >> count;
    direct_model.bigd_table = read_doubles(in, count);
    expect_token(in, "p_table");
    in >> count;
    direct_model.p_table = read_doubles(in, count);
    expect_token(in, "embedding");
    in >> count;
    direct_model.embedding = read_doubles(in, count);

    auto read_layers = [&](std::vector<DenseLayer>& layers, std::vector<Activation>& activations,
                           const std::string& token) {
      int nlayer = 0;
      expect_token(in, token);
      in >> nlayer;
      layers.clear();
      activations.clear();
      layers.reserve(nlayer);
      activations.reserve(nlayer);

      for (int ilayer = 0; ilayer < nlayer; ++ilayer) {
        DenseLayer layer;
        std::string activation;
        expect_token(in, "layer");
        in >> layer.out >> layer.in >> layer.has_bias >> activation;
        layer.weight = read_doubles(in, static_cast<size_t>(layer.out) * layer.in);
        if (layer.has_bias) layer.bias = read_doubles(in, layer.out);
        else layer.bias.assign(layer.out, 0.0);

        if (activation == "linear") activations.push_back(Activation::LINEAR);
        else if (activation == "silu") activations.push_back(Activation::SILU);
        else if (activation == "softplus") activations.push_back(Activation::SOFTPLUS);
        else if (activation == "shifted_softplus") activations.push_back(Activation::SHIFTED_SOFTPLUS);
        else throw std::runtime_error("Unknown activation: " + activation);

        layers.push_back(std::move(layer));
      }
    };

    read_layers(direct_model.xi_layers, direct_model.xi_activations, "xi_layers");
    read_layers(direct_model.bij_layers, direct_model.bij_activations, "bij_layers");

    expect_token(in, "end");
    direct_model_loaded = true;

    if (comm->me == 0){
      std::cout << "Direct NN model loaded successfully from: " << model_path << std::endl;
      std::cout << "Using pair cutoff function: "
                << direct_model.pair_cutoff_function_name << std::endl;
      std::cout << "Using NN cutoff function: "
                << direct_model.nn_cutoff_function_name << std::endl;
    }
  }
  catch (const std::exception& e) {
    error->all(FLERR, "Error loading direct NN model '{}': {}", model_path, e.what());
  }
}

void PairTersoffNNDirect::load_csv(const std::string& csv_path) {
  std::ifstream ifs(csv_path);
  if (!ifs) {
    error->all(FLERR, "Cannot open input statistics file '{}'", csv_path);
  }
  stats_map.clear();
  while (std::getline(ifs, line_csv)) {
    if (line_csv.empty()) continue;

    std::stringstream ss(line_csv);
    std::getline(ss, feature, ',');
    std::getline(ss, min_str, ',');
    std::getline(ss, max_str, ',');

    if (feature.empty() || min_str.empty() || max_str.empty()) {
      continue;
    }
    if (feature == "feature" || min_str == "min" || max_str == "max") {
      continue;
    }

    min_val = std::stod(min_str);
    max_val = std::stod(max_str);
    stats_map[feature] = std::make_pair(min_val, max_val);
  }
  ifs.close();

  std::array<std::string, 6> feature_names = {"rij", "rik", "cosjik", "type_i", "type_j", "type_k"};
  for (size_t i = 0; i < feature_names.size(); i++) {
    const std::string& fname = feature_names[i];
    if (stats_map.find(fname) == stats_map.end()) {
      throw std::runtime_error("Feature not found in statistics: " + fname);
    }
    auto [min_val, max_val] = stats_map[fname];
    feature_mins[i] = min_val;
    feature_maxs[i] = max_val;
  }
  calc_scaling_factors();
}

void PairTersoffNNDirect::calc_scaling_factors(){
  double scaling_max = 1.0;
  double scaling_min = 0.0;
  std::array<std::string, 6> feature_names = {"rij", "rik", "cosjik", "type_i", "type_j", "type_k"};

  for (size_t i = 0; i < feature_names.size(); i++) {
      const auto& fname = feature_names[i];
      if (stats_map.find(fname) == stats_map.end()) {
          throw std::runtime_error("Feature not found in statistics: " + fname);
      }
      auto [min_val, max_val] = stats_map[fname];
      if (std::abs(max_val - min_val) < 1e-12) {
          throw std::runtime_error(
              "Cannot compute scaling factor because max_val == min_val for feature: " + fname
          );
      }
      scaling_factors[i] = (scaling_max - scaling_min) / (max_val - min_val);
  }
}

void PairTersoffNNDirect::nn_attractive_reg(Param *param, double dbij_drij,
                                  double rsqij, double *rij_hat,
                                  double *fi, double *fj, int eflag,
                                  double &eng, double &fforce)
{
  double rij,fa,fc,dfa,dfc;

  rij = sqrt(rsqij);

  // fa and dfa exclude fc so the cutoff remains explicit in each force term.
  fa = nn_ters_fa(rij,param);
  dfa = nn_ters_fa_d(rij,param);
  fc = ters_fc(rij,param);
  dfc = ters_fc_d(rij,param);

  scale3(0.5*dfc*b_ij*fa,rij_hat,fi);
  scaleadd3(0.5*fc*dbij_drij*fa,rij_hat,fi,fi);
  scaleadd3(0.5*fc*b_ij*dfa,rij_hat,fi,fi);

  scale3(-0.5*dfc*b_ij*fa,rij_hat,fj);
  scaleadd3(-0.5*fc*dbij_drij*fa,rij_hat,fj,fj);
  scaleadd3(-0.5*fc*b_ij*dfa,rij_hat,fj,fj);

  fforce = 0.5*dfc*b_ij*fa+0.5*fc*dbij_drij*fa+0.5*fc*b_ij*dfa;
  if (eflag) {
    eng = 0.5*fc*b_ij*fa;
    //std::cout << eng << std::endl;
  }

}

void PairTersoffNNDirect::nn_attractive_ex(double attractive_scale,
                                           double dbij_drik, double dbij_drjk,
                                           double *rik_hat, double* rjk_hat,
                                           double *fi, double *fj, double *fk)
{
  const double dE_attractive_drik = attractive_scale * dbij_drik;
  const double dE_attractive_drjk = attractive_scale * dbij_drjk;
  scale3(dE_attractive_drik,rik_hat,fi);
  scale3(-dE_attractive_drik,rik_hat,fk);
  scale3(dE_attractive_drjk,rjk_hat,fj);
  scaleadd3(-dE_attractive_drjk,rjk_hat,fk,fk);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNNDirect::nn_ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNNDirect::nn_ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) * param->lam2;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::eval_xi_batch_with_grad(int batch_size)
{
  const int input_size = 3 + 3 * direct_model.dim_embed_per_atom;
  eval_mlp_batch_with_grad(direct_model.xi_layers,
                           direct_model.xi_activations,
                           xi_batch_inputs_.data(),
                           batch_size,
                           input_size,
                           xi_values_.data(),
                           xi_grads_.data(),
                           xi_mlp_workspace_);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::eval_bij_batch_with_grad(int batch_size)
{
  eval_mlp_batch_with_grad(direct_model.bij_layers,
                           direct_model.bij_activations,
                           bij_batch_inputs_.data(),
                           batch_size,
                           1,
                           bij_values_.data(),
                           bij_grads_.data(),
                           bij_mlp_workspace_);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNNDirect::eval_mlp_batch_with_grad(const std::vector<DenseLayer>& layers,
                                                   const std::vector<Activation>& activations,
                                                   const double* inputs,
                                                   int batch_size,
                                                   int input_size,
                                                   double* outputs,
                                                   double* grad_inputs,
                                                   MLPBatchWorkspace& workspace) const
{
  if (batch_size <= 0) return;
  if (layers.empty()) throw std::runtime_error("Direct NN expects at least one layer");
  if (layers.size() != activations.size()) {
    throw std::runtime_error("Direct NN layer/activation mismatch");
  }

  std::vector<int>& dims = workspace.dims;
  dims.resize(layers.size() + 1);
  dims[0] = input_size;
  for (size_t ilayer = 0; ilayer < layers.size(); ++ilayer) {
    const DenseLayer& layer = layers[ilayer];
    if (layer.in != dims[ilayer]) {
      throw std::runtime_error("Direct NN layer dimension mismatch");
    }
    dims[ilayer + 1] = layer.out;
  }
  if (dims.back() != 1) throw std::runtime_error("Direct NN expects scalar output");

  std::vector<std::vector<double>>& values = workspace.values;
  std::vector<std::vector<double>>& act_derivs = workspace.act_derivs;
  values.resize(layers.size() + 1);
  act_derivs.resize(layers.size());
  values[0].resize(static_cast<size_t>(input_size) * batch_size);

  for (int b = 0; b < batch_size; ++b) {
    const double *input = inputs + static_cast<size_t>(b) * input_size;
    for (int i = 0; i < input_size; ++i) {
      values[0][static_cast<size_t>(i) * batch_size + b] = input[i];
    }
  }

  for (size_t ilayer = 0; ilayer < layers.size(); ++ilayer) {
    const DenseLayer& layer = layers[ilayer];
    const double *prev = values[ilayer].data();
    std::vector<double>& current = values[ilayer + 1];
    std::vector<double>& deriv = act_derivs[ilayer];
    current.resize(static_cast<size_t>(layer.out) * batch_size);
    deriv.resize(static_cast<size_t>(layer.out) * batch_size);

    for (int o = 0; o < layer.out; ++o) {
      double *out_vec = current.data() + static_cast<size_t>(o) * batch_size;
      double *deriv_vec = deriv.data() + static_cast<size_t>(o) * batch_size;
      std::fill(out_vec, out_vec + batch_size, layer.bias[o]);
      const size_t row = static_cast<size_t>(o) * layer.in;

      for (int in_idx = 0; in_idx < layer.in; ++in_idx) {
        const double weight = layer.weight[row + in_idx];
        const double *in_vec = prev + static_cast<size_t>(in_idx) * batch_size;
        for (int b = 0; b < batch_size; ++b) {
          out_vec[b] += weight * in_vec[b];
        }
      }

      const Activation activation = activations[ilayer];
      if (activation == Activation::LINEAR) {
        for (int b = 0; b < batch_size; ++b) {
          deriv_vec[b] = 1.0;
        }
      } else if (activation == Activation::SILU) {
        for (int b = 0; b < batch_size; ++b) {
          const double zval = out_vec[b];
          const double sig = 1.0 / (1.0 + std::exp(-zval));
          out_vec[b] = zval * sig;
          deriv_vec[b] = sig * (1.0 + zval * (1.0 - sig));
        }
      } else if (activation == Activation::SOFTPLUS ||
                 activation == Activation::SHIFTED_SOFTPLUS) {
        const double shift_value =
            (activation == Activation::SHIFTED_SOFTPLUS) ? std::log(2.0) : 0.0;
        for (int b = 0; b < batch_size; ++b) {
          const double zval = out_vec[b];
          double value;
          double sig;
          if (zval > 40.0) {
            value = zval;
            sig = 1.0;
          } else if (zval < -40.0) {
            value = std::exp(zval);
            sig = value;
          } else {
            const double exp_z = std::exp(zval);
            value = std::log1p(exp_z);
            sig = exp_z / (1.0 + exp_z);
          }
          out_vec[b] = value - shift_value;
          deriv_vec[b] = sig;
        }
      }
    }
  }

  const double *final_values = values.back().data();
  for (int b = 0; b < batch_size; ++b) outputs[b] = final_values[b];

  std::vector<double>& delta = workspace.delta;
  std::vector<double>& prev_delta = workspace.prev_delta;
  delta.assign(static_cast<size_t>(dims.back()) * batch_size, 1.0);

  for (int ilayer = static_cast<int>(layers.size()) - 1; ilayer >= 0; --ilayer) {
    const DenseLayer& layer = layers[ilayer];
    const std::vector<double>& deriv = act_derivs[ilayer];

    for (int o = 0; o < layer.out; ++o) {
      double *delta_vec = delta.data() + static_cast<size_t>(o) * batch_size;
      const double *deriv_vec = deriv.data() + static_cast<size_t>(o) * batch_size;
      for (int b = 0; b < batch_size; ++b) {
        delta_vec[b] *= deriv_vec[b];
      }
    }

    prev_delta.assign(static_cast<size_t>(layer.in) * batch_size, 0.0);
    for (int o = 0; o < layer.out; ++o) {
      const double *delta_vec = delta.data() + static_cast<size_t>(o) * batch_size;
      const size_t row = static_cast<size_t>(o) * layer.in;
      for (int in_idx = 0; in_idx < layer.in; ++in_idx) {
        const double weight = layer.weight[row + in_idx];
        double *prev_vec = prev_delta.data() + static_cast<size_t>(in_idx) * batch_size;
        for (int b = 0; b < batch_size; ++b) {
          prev_vec[b] += weight * delta_vec[b];
        }
      }
    }

    delta.swap(prev_delta);
  }

  for (int b = 0; b < batch_size; ++b) {
    double *grad_input = grad_inputs + static_cast<size_t>(b) * input_size;
    for (int i = 0; i < input_size; ++i) {
      grad_input[i] = delta[static_cast<size_t>(i) * batch_size + b];
    }
  }
}

int PairTersoffNNDirect::feature_type_to_index(double value, int feature_idx) const
{
  const double scaled = (value - feature_mins[feature_idx]) * scaling_factors[feature_idx];
  int idx = static_cast<int>(std::floor(scaled + 1.0e-6));
  if (idx < 0) idx = 0;
  if (direct_model.num_types > 0 && idx >= direct_model.num_types) idx = direct_model.num_types - 1;
  return idx;
}

int PairTersoffNNDirect::elem_params_manual(int itype, int jtype, int ktype) const
{
  return direct_model.num_types * direct_model.num_types * itype
       + direct_model.num_types * jtype + ktype;
}

int PairTersoffNNDirect::elem_params_pair_manual(int itype, int jtype) const
{
  const int ii = std::min(itype, jtype);
  const int jj = std::max(itype, jtype);
  return direct_model.num_types * ii + jj;
}

double PairTersoffNNDirect::fc_manual(double r, int itype, int jtype, int ktype) const
{
  const int p = elem_params_manual(itype, jtype, ktype);
  const double bigr = direct_model.bigr_table[p];
  const double bigd = direct_model.bigd_table[p];
  const double x1 = bigr - bigd;
  const double x2 = bigr + bigd;
  if (r < x1) return 1.0;
  if (r > x2) return 1.0e-7;

  const double phase = MY_PI * (r - bigr) / (2.0 * bigd);
  if (direct_model.nn_cutoff_function == CutoffFunction::SMOOTH)
    return 0.5 - 0.5625 * std::sin(phase) - 0.0625 * std::sin(3.0 * phase);

  return 0.5 * (1.0 - std::sin(phase));
}

double PairTersoffNNDirect::fc_manual_d(double r, int itype, int jtype, int ktype) const
{
  const int p = elem_params_manual(itype, jtype, ktype);
  const double bigr = direct_model.bigr_table[p];
  const double bigd = direct_model.bigd_table[p];
  const double x1 = bigr - bigd;
  const double x2 = bigr + bigd;
  if (r < x1 || r > x2) return 0.0;

  const double dphase = MY_PI / (2.0 * bigd);
  const double phase = dphase * (r - bigr);
  if (direct_model.nn_cutoff_function == CutoffFunction::SMOOTH)
    return -dphase *
      (0.5625 * std::cos(phase) + 0.1875 * std::cos(3.0 * phase));

  return -0.5 * dphase * std::cos(phase);
}
