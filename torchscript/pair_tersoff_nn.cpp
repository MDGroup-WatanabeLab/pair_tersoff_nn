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
                        Kotaro Takematsu (Waseda University) - initial Tersoff-NN implementation
                        Yusuke Nishimura (Waseda University) - Tersoff-NN optimization and bug fix
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Derived from LAMMPS src/MANYBODY/pair_tersoff.cpp and modified for
   Tersoff-NN bond-order evaluation through a TorchScript model and LibTorch.
------------------------------------------------------------------------- */

#include "pair_tersoff_nn.h"
#include <torch/cuda.h>
#include <torch/script.h>

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

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

static constexpr int DELTA = 4;

/* ---------------------------------------------------------------------- */

PairTersoffNN::PairTersoffNN(LAMMPS *lmp) : Pair(lmp)
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

PairTersoffNN::~PairTersoffNN()
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

void PairTersoffNN::compute(int eflag, int vflag)
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
void PairTersoffNN::eval()
{
  int i,j,k,ii,jj,kk,inum,jnum;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double fforce;
  double rsq,rsq1,rsq2,rsq3;
  double delr1[3],delr2[3],delr3[3],fi[3],fj[3],fk[3];
  double r1_hat[3],r2_hat[3],r3_hat[3];
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

  // prepare NN input and mask
  torch::Tensor nn_input = torch::zeros({inum, max_pair, max_triplet, 6}, torch::kFloat32);
  torch::Tensor pair_mask = torch::zeros({inum, max_pair}, torch::kBool);
  torch::Tensor triplet_mask = torch::zeros({inum, max_pair, max_triplet}, torch::kBool);

  if (use_gpu_ && torch::cuda::is_available()) {
    nn_input = nn_input.to(torch::kCUDA);
    pair_mask = pair_mask.to(torch::kCUDA);
    triplet_mask = triplet_mask.to(torch::kCUDA);
  }

  auto nn_input_acc = nn_input.accessor<float, 4>();
  auto pm_acc = pair_mask.accessor<bool, 2>();
  auto tm_acc = triplet_mask.accessor<bool, 3>();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    const int start = short_ofs[ii];
    const int stop  = short_ofs[ii+1];
    const int numshort = stop - start;

    // store input per atom to accessors
    int pair_idx = 0;

    for (jj = 0; jj < numshort && pair_idx < max_pair; jj++) {
      j = short_buf[start + jj];
      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];

      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      double rsq1_eff = rsq1;

      if (SHIFT_FLAG) rsq1_eff += shift*shift + 2*sqrt(rsq1)*shift;
      if (rsq1_eff >= params[iparam_ij].cutsq) continue;

      const double r1inv = 1.0/sqrt(dot3(delr1, delr1));
      scale3(r1inv, delr1, r1_hat);

      int t = 0; // triplet index
      for (kk = 0; kk < numshort && t < max_triplet; kk++) {
        if (jj == kk) continue;
        k = short_buf[start + kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        // std::cout << iparam_ijk << std::endl;

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        double rsq2_eff = rsq2;

        if (SHIFT_FLAG) rsq2_eff += shift*shift + 2*sqrt(rsq2)*shift;
        if (rsq2_eff >= params[iparam_ijk].cutsq) continue;

        double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        // features
        const float geo1 = static_cast<float>(sqrt(rsq1)); // rij
        const float geo2 = static_cast<float>(sqrt(rsq2)); // rik
        const float geo3 = static_cast<float>(dot3(r1_hat, r2_hat)); // cosjik
        const float geo4 = static_cast<float>(itype); // type_i
        const float geo5 = static_cast<float>(jtype); // type_j
        const float geo6 = static_cast<float>(ktype); // type_k

        nn_input_acc[ii][pair_idx][t][0] = geo1;
        nn_input_acc[ii][pair_idx][t][1] = geo2;
        nn_input_acc[ii][pair_idx][t][2] = geo3;
        nn_input_acc[ii][pair_idx][t][3] = geo4;
        nn_input_acc[ii][pair_idx][t][4] = geo5;
        nn_input_acc[ii][pair_idx][t][5] = geo6;

        tm_acc[ii][pair_idx][t] = true;
        ++t;
      }

      if (t > 0) {
        pm_acc[ii][pair_idx] = true;
        ++pair_idx;
      }
    }
  }

  auto nn_input_scaled = scale_tensor_minmax(nn_input);

  ters_nn_bij_update(nn_input_scaled, pair_mask, triplet_mask);

  // calculate energy, force, stress
  auto grad_acc = grad_original.accessor<float, 4>();
  auto b_ij_acc = b_ij_list.accessor<float, 2>();

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

    double fjxtmp,fjytmp,fjztmp;
    int pair_idx = 0;

    // 3 body interaction (attractive)
    for (jj = 0; jj < numshort; jj++) {
      j = short_buf[start + jj];
      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];

      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      if (SHIFT_FLAG)
        rsq1 += shift*shift + 2*sqrt(rsq1)*shift;

      if (rsq1 >= params[iparam_ij].cutsq) continue;

      const double r1inv = 1.0/sqrt(dot3(delr1, delr1));
      scale3(r1inv, delr1, r1_hat);

      // accumulate bondorder zeta for each i-j interaction via loop over k

      fjxtmp = fjytmp = fjztmp = 0.0;

      // attractive term via loop over k
      // calc dbij / drij

      double dbijdrij = 0.0;
      int trip_idx = 0;
      bool had_trip = false;

      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;

        k = short_buf[start + kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        if (SHIFT_FLAG)
          rsq2 += shift*shift + 2*sqrt(rsq2)*shift;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        had_trip = true;

        double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        delr3[0] = x[k][0] - x[j][0];
        delr3[1] = x[k][1] - x[j][1];
        delr3[2] = x[k][2] - x[j][2];
        rsq3 = delr3[0]*delr3[0] + delr3[1]*delr3[1] + delr3[2]*delr3[2];

        double r3inv = 1.0/sqrt(dot3(delr3, delr3));
        scale3(r3inv, delr3, r3_hat);

        double dbijdrik = 0.0;
        double dbijdrjk = 0.0;

        // this part will never called when triplets == 0

        const double g_rij = static_cast<double>(grad_acc[ii][pair_idx][trip_idx][0]);
        const double g_rik = static_cast<double>(grad_acc[ii][pair_idx][trip_idx][1]);
        const double g_cos = static_cast<double>(grad_acc[ii][pair_idx][trip_idx][2]);

        const double dcosjikdrij = (rsq1 - rsq2 + rsq3) / (2 * rsq1 * sqrt(rsq2));
        const double dcosjikdrik = (-rsq1 + rsq2 + rsq3) / (2 * rsq2 * sqrt(rsq1));
        const double dcosjikdrjk = - sqrt(rsq3) / (sqrt(rsq1) * sqrt(rsq2));

        dbijdrij += g_rij + g_cos * dcosjikdrij;
        dbijdrik += g_rik + g_cos * dcosjikdrik;
        dbijdrjk +=         g_cos * dcosjikdrjk;

        nn_attractive_ex(&params[iparam_ij],dbijdrik,dbijdrjk,rsq1,rsq2,rsq3,r1_hat,r2_hat,r3_hat,fi,fj,fk);

        if (VFLAG_EITHER) v_tally3(i,j,k,fj,fk,delr1,delr2);

        fxtmp += fi[0];
        fytmp += fi[1];
        fztmp += fi[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        ++trip_idx;
      }

      if (!had_trip){
        b_ij = 1.0;
      } else {
        b_ij = static_cast<double>(b_ij_acc[ii][pair_idx]);
        ++pair_idx;
      }
      //std::cout << "count for grad" << count << std::endl;
      //if (triplets < triplet_count_check) continue;
      nn_attractive_reg(&params[iparam_ij],dbijdrij,rsq1,rsq2,rsq3,r1_hat,r2_hat,r3_hat,fi,fj,fk,EFLAG,evdwl,fforce);

      fpair = fforce*r1inv;

      if (EVFLAG) ev_tally(i,j,nlocal,newton_pair,
                          evdwl,0.0,-fpair,-delr1[0],-delr1[1],-delr1[2]);

      fxtmp += fi[0];
      fytmp += fi[1];
      fztmp += fi[2];
      fjxtmp += fj[0];
      fjytmp += fj[1];
      fjztmp += fj[2];

      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }

    //std::cout << "exit" << std::endl;
    //std::exit(0);

    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
    //count_ii_loop += 1;
  }
  if (vflag_fdotr) virial_fdotr_compute();
  //std::exit(0);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::allocate()
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

void PairTersoffNN::settings(int narg, char **arg)
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
        use_gpu_ = true;
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

  if (use_gpu_ && !torch::cuda::is_available()) {
    error->all(FLERR,"use_gpu yes was requested, but CUDA is not available to LibTorch");
  }
  if (!input_stats) error->all(FLERR,"input_stats is required for tersoff_nn");
  if (!model_name)  error->all(FLERR,"model_name is required for tersoff_nn");
  load_csv(input_stats);
  load_model(model_name);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTersoffNN::coeff(int narg, char **arg)
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

void PairTersoffNN::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Tersoff_nn requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Tersoff_nn requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this,NeighConst::REQ_FULL);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTersoffNN::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status\n" + Info::get_pair_coeff_status(lmp));

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::read_file(char *file)
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

void PairTersoffNN::setup_params()
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

void PairTersoffNN::repulsive(Param *param, double rsq, double &fforce,
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

double PairTersoffNN::zeta(Param *param, double rsqij, double rsqik,
                         double *rij_hat, double *rik_hat)
{
  double rij,rik,costheta,arg,ex_delr;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  costheta = dot3(rij_hat,rik_hat);

  if (param->powermint == 3) arg = cube(param->lam3 * (rij-rik));
  else arg = param->lam3 * (rij-rik);

  if (arg > 69.0776) ex_delr = 1.e30;
  else if (arg < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(arg);

  return ters_fc(rik,param) * ters_gijk(costheta,param) * ex_delr;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::force_zeta(Param *param, double rsq, double zeta_ij,
                             double &fforce, double &prefactor,
                             int eflag, double &eng)
{
  double r,fa,fa_d,bij;

  r = sqrt(rsq);
  fa = ters_fa(r,param);
  fa_d = ters_fa_d(r,param);
  bij = ters_bij(zeta_ij,param);
  fforce = 0.5*bij*fa_d;
  prefactor = -0.5*fa * ters_bij_d(zeta_ij,param);
  if (eflag) eng = 0.5*bij*fa;
}

/* ----------------------------------------------------------------------
   attractive term
   use param_ij cutoff for rij test
   use param_ijk cutoff for rik test
------------------------------------------------------------------------- */

void PairTersoffNN::attractive(Param *param, double prefactor,
                             double rsqij, double rsqik,
                             double *rij_hat, double *rik_hat,
                             double *fi, double *fj, double *fk)
{
  double rij,rijinv,rik,rikinv;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);

  // correct 1/r for shift in rsq

  if (shift_flag == 1) {
    rijinv = 1.0/(rij - shift);
    rikinv = 1.0/(rik - shift);
  } else {
    rijinv = 1.0/rij;
    rikinv = 1.0/rik;
  }

  ters_zetaterm_d(prefactor,rij_hat,rij,rijinv,rik_hat,rik,rikinv,fi,fj,fk,param);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_fc(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 1.0;
  if (r > ters_R+ters_D) return 0.0;

  const double phase = MY_PI2*(r - ters_R)/ters_D;
  if (cutoff_function_ == CutoffFunction::SMOOTH)
    return 0.5 - 0.5625*sin(phase) - 0.0625*sin(3.0*phase);

  return 0.5*(1.0 - sin(phase));
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_fc_d(double r, Param *param)
{
  const double ters_R = param->bigr;
  const double ters_D = param->bigd;

  if (r < ters_R-ters_D) return 0.0;
  if (r > ters_R+ters_D) return 0.0;

  const double phase = MY_PI2*(r - ters_R)/ters_D;
  if (cutoff_function_ == CutoffFunction::SMOOTH)
    return -(MY_PI2/ters_D) *
      (0.5625*cos(phase) + 0.1875*cos(3.0*phase));

  return -(MY_PI4/ters_D) * cos(phase);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r) * ters_fc(r,param);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) *
    (param->lam2 * ters_fc(r,param) - ters_fc_d(r,param));
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_bij(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  if (tmp > param->c1) return 1.0/sqrt(tmp);
  if (tmp > param->c2)
    return (1.0 - pow(tmp,-param->powern) / (2.0*param->powern))/sqrt(tmp);
  if (tmp < param->c4) return 1.0;
  if (tmp < param->c3)
    return 1.0 - pow(tmp,param->powern)/(2.0*param->powern);
  return pow(1.0 + pow(tmp,param->powern), -1.0/(2.0*param->powern));
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::ters_bij_d(double zeta, Param *param)
{
  double tmp = param->beta * zeta;
  if (tmp > param->c1) return param->beta * -0.5*pow(tmp,-1.5);
  if (tmp > param->c2)
    return param->beta * (-0.5*pow(tmp,-1.5) *
                          // error in negligible 2nd term fixed 9/30/2015
                          // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                          (1.0 - (1.0 +  1.0/(2.0*param->powern)) *
                           pow(tmp,-param->powern)));
  if (tmp < param->c4) return 0.0;
  if (tmp < param->c3)
    return -0.5*param->beta * pow(tmp,param->powern-1.0);

  double tmp_n = pow(tmp,param->powern);
  return -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*param->powern)))*tmp_n / zeta;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::ters_zetaterm_d(double prefactor,
                                  double *rij_hat, double rij, double rijinv,
                                  double *rik_hat, double rik, double rikinv,
                                  double *dri, double *drj, double *drk,
                                  Param *param)
{
  double gijk,gijk_d,ex_delr,ex_delr_d,fc,dfc,cos_theta,tmp;
  double dcosdri[3],dcosdrj[3],dcosdrk[3];

  fc = ters_fc(rik,param);
  dfc = ters_fc_d(rik,param);
  if (param->powermint == 3) tmp = cube(param->lam3 * (rij-rik));
  else tmp = param->lam3 * (rij-rik);

  if (tmp > 69.0776) ex_delr = 1.e30;
  else if (tmp < -69.0776) ex_delr = 0.0;
  else ex_delr = exp(tmp);

  if (param->powermint == 3)
    ex_delr_d = 3.0*cube(param->lam3) * square(rij-rik)*ex_delr;
  else ex_delr_d = param->lam3 * ex_delr;

  cos_theta = dot3(rij_hat,rik_hat);
  gijk = ters_gijk(cos_theta,param);
  gijk_d = ters_gijk_d(cos_theta,param);
  costheta_d(rij_hat,rijinv,rik_hat,rikinv,dcosdri,dcosdrj,dcosdrk);

  // compute the derivative wrt Ri
  // dri = -dfc*gijk*ex_delr*rik_hat;
  // dri += fc*gijk_d*ex_delr*dcosdri;
  // dri += fc*gijk*ex_delr_d*(rik_hat - rij_hat);

  scale3(-dfc*gijk*ex_delr,rik_hat,dri);
  scaleadd3(fc*gijk_d*ex_delr,dcosdri,dri,dri);
  scaleadd3(fc*gijk*ex_delr_d,rik_hat,dri,dri);
  scaleadd3(-fc*gijk*ex_delr_d,rij_hat,dri,dri);
  scale3(prefactor,dri);

  // compute the derivative wrt Rj
  // drj = fc*gijk_d*ex_delr*dcosdrj;
  // drj += fc*gijk*ex_delr_d*rij_hat;

  scale3(fc*gijk_d*ex_delr,dcosdrj,drj);
  scaleadd3(fc*gijk*ex_delr_d,rij_hat,drj,drj);
  scale3(prefactor,drj);

  // compute the derivative wrt Rk
  // drk = dfc*gijk*ex_delr*rik_hat;
  // drk += fc*gijk_d*ex_delr*dcosdrk;
  // drk += -fc*gijk*ex_delr_d*rik_hat;

  scale3(dfc*gijk*ex_delr,rik_hat,drk);
  scaleadd3(fc*gijk_d*ex_delr,dcosdrk,drk,drk);
  scaleadd3(-fc*gijk*ex_delr_d,rik_hat,drk,drk);
  scale3(prefactor,drk);
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::costheta_d(double *rij_hat, double rijinv,
                             double *rik_hat, double rikinv,
                             double *dri, double *drj, double *drk)
{
  // first element is devative wrt Ri, second wrt Rj, third wrt Rk

  double cos_theta = dot3(rij_hat,rik_hat);

  scaleadd3(-cos_theta,rij_hat,rik_hat,drj);
  scale3(rijinv,drj);
  scaleadd3(-cos_theta,rik_hat,rij_hat,drk);
  scale3(rikinv,drk,drk);
  add3(drj,drk,dri);
  scale3(-1.0,dri);
}

void PairTersoffNN::load_model(const std::string& model_path) {
  try {
    tersoff_nn_model = torch::jit::load(model_path);
    load_cutoff_function_from_model();
    if (use_gpu_ && torch::cuda::is_available()) {
      tersoff_nn_model.to(torch::kCUDA);
    }
    tersoff_nn_model.eval();
    if (comm->me == 0){
      std::cout << "Model loaded successfully from: " << model_path << std::endl;
    }
    for (const auto& param :tersoff_nn_model.parameters()) {
    }
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "Error loading TorchScript model '{}': {}", model_path, e.what());
  }
}

void PairTersoffNN::load_cutoff_function_from_model()
{
  const auto cutoff_method = tersoff_nn_model.find_method("get_cutoff_function");
  if (!cutoff_method) {
    cutoff_function_ = CutoffFunction::TERSOFF;
    cutoff_function_name_ = "tersoff";
    if (comm->me == 0) {
      std::cout << "Model does not store cutoff_function; falling back to tersoff"
                << std::endl;
    }
    return;
  }

  cutoff_function_name_ = (*cutoff_method)({}).toStringRef();
  if (cutoff_function_name_ == "tersoff") {
    cutoff_function_ = CutoffFunction::TERSOFF;
  } else if (cutoff_function_name_ == "smooth") {
    cutoff_function_ = CutoffFunction::SMOOTH;
  } else {
    error->all(FLERR, "Unsupported cutoff_function '{}' stored in model",
               cutoff_function_name_);
  }

  if (comm->me == 0) {
    std::cout << "Using cutoff_function from model: "
              << cutoff_function_name_ << std::endl;
  }
}

void PairTersoffNN::load_csv(const std::string& csv_path) {
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
  std::array<float, 6> feature_mins;
  std::array<float, 6> feature_maxs;
  for (size_t i = 0; i < feature_names.size(); i++) {
    const std::string& fname = feature_names[i];
    if (stats_map.find(fname) == stats_map.end()) {
      throw std::runtime_error("Feature not found in statistics: " + fname);
    }
    auto [min_val, max_val] = stats_map[fname];
    feature_mins[i] = static_cast<float>(min_val);
    feature_maxs[i] = static_cast<float>(max_val);
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  t_min = torch::from_blob(feature_mins.data(), {(long)feature_mins.size()}, options).clone();
  t_max = torch::from_blob(feature_maxs.data(), {(long)feature_maxs.size()}, options).clone();

  if (use_gpu_ && torch::cuda::is_available()) {
    t_min = t_min.to(torch::kCUDA);
    t_max = t_max.to(torch::kCUDA);
  }
  calc_scaling_factors();
}

void PairTersoffNN::calc_scaling_factors(){
  double scaling_max = 1.0;
  double scaling_min = 0.0;
  std::array<std::string, 6> feature_names = {"rij", "rik", "cosjik", "type_i", "type_j", "type_k"};
  scaling_factors = torch::empty({6}, torch::kFloat32);

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
      //scaling_factors[i] = (scaling_max - scaling_min) / (max_val - min_val);
      scaling_factors[i] = (scaling_max - scaling_min) / (max_val - min_val);
  }
}

torch::Tensor PairTersoffNN::scale_tensor_minmax(const torch::Tensor& tensor) {
  return (tensor - t_min) / (t_max - t_min);
}

void PairTersoffNN::nn_attractive_reg(Param *param, double dbij_drij,
                                  double rsqij, double rsqik, double rsqjk,
                                  double *rij_hat, double *rik_hat, double* rjk_hat,
                                  double *fi, double *fj, double *fk, int eflag, double &eng, double &fforce)
{
  double rij,rik,rjk,fa,fc,dfa,dfc;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  rjk = sqrt(rsqjk);

  //fa = ters_fa(rij,param);
  fa = nn_ters_fa(rij,param);
  //dfa = ters_fa_d(rij,param);
  dfa = nn_ters_fa_d(rij,param);
  fc = ters_fc(rij,param);
  dfc = ters_fc_d(rij,param);

  scale3(0.5*dfc*b_ij*fa,rij_hat,fi);
  scaleadd3(0.5*fc*dbij_drij*fa,rij_hat,fi,fi);
  scaleadd3(0.5*fc*b_ij*dfa,rij_hat,fi,fi);

  scale3(-0.5*dfc*b_ij*fa,rij_hat,fj);
  scaleadd3(-0.5*fc*dbij_drij*fa,rij_hat,fj,fj);
  scaleadd3(-0.5*fc*b_ij*dfa,rij_hat,fj,fj);
  /*
  scale3(dfc*b_ij*fa,rij_hat,fi);
  scaleadd3(fc*dbij_drij*fa,rij_hat,fi,fi);
  scaleadd3(fc*b_ij*dfa,rij_hat,fi,fi);

  scale3(-dfc*b_ij*fa,rij_hat,fj);
  scaleadd3(-fc*dbij_drij*fa,rij_hat,fj,fj);
  scaleadd3(-fc*b_ij*dfa,rij_hat,fj,fj);
  */
  //fforce = -0.5*dfc*b_ij*fa-0.5*fc*dbij_drij*fa-0.5*fc*b_ij*dfa;
  fforce = 0.5*dfc*b_ij*fa+0.5*fc*dbij_drij*fa+0.5*fc*b_ij*dfa;
  if (eflag) {
    eng = 0.5*fc*b_ij*fa;
    //std::cout << eng << std::endl;
  }

}

void PairTersoffNN::nn_attractive_ex(Param *param, double dbij_drik, double dbij_drjk,
                                  double rsqij, double rsqik, double rsqjk,
                                  double *rij_hat, double *rik_hat, double* rjk_hat,
                                  double *fi, double *fj, double *fk)
{
  double rij,rik,rjk,fa,fc;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  rjk = sqrt(rsqjk);

  fa = nn_ters_fa(rij,param);
  fc = ters_fc(rij,param);

  scale3(0.5*fc*dbij_drik*fa,rik_hat,fi);
  scale3(-0.5*fc*dbij_drik*fa,rik_hat,fk);
  scale3(0.5*fc*dbij_drjk*fa,rjk_hat,fj);
  scaleadd3(-0.5*fc*dbij_drjk*fa,rjk_hat,fk,fk);
  /*
  scale3(fc*dbij_drik*fa,rik_hat,fi);
  scale3(-fc*dbij_drik*fa,rik_hat,fk);
  scale3(fc*dbij_drjk*fa,rjk_hat,fj);
  scaleadd3(-fc*dbij_drjk*fa,rjk_hat,fk,fk);
  */
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::nn_ters_fa(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return -param->bigb * exp(-param->lam2 * r);
}

/* ---------------------------------------------------------------------- */

double PairTersoffNN::nn_ters_fa_d(double r, Param *param)
{
  if (r > param->bigr + param->bigd) return 0.0;
  return param->bigb * exp(-param->lam2 * r) * param->lam2;
}

/* ---------------------------------------------------------------------- */

void PairTersoffNN::ters_nn_bij_update(const torch::Tensor& scaled_input,
                                      const torch::Tensor& pair_mask,
                                      const torch::Tensor& triplet_mask){

  torch::Tensor x = scaled_input.clone().detach().set_requires_grad(true);
  torch::Tensor pm = pair_mask;
  torch::Tensor tm = triplet_mask;

  if (use_gpu_ && torch::cuda::is_available()) {
    x = x.to(torch::kCUDA);
    pm = pm.to(torch::kCUDA);
    tm = tm.to(torch::kCUDA);
  }

  auto output = tersoff_nn_model.forward({x, pm, tm}).toTensor();
  //py::gil_scoped_release no_gil;
  output.sum().backward();

  torch::Tensor grad_cpu = x.grad().to(torch::kCPU).detach().contiguous();
  torch::Tensor output_cpu = output.detach().to(torch::kCPU).contiguous();
  torch::Tensor pm_cpu = pair_mask.to(torch::kCPU);

  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor b_full = torch::zeros(pm_cpu.sizes(), opts_f32);

  b_full = b_full.masked_scatter(pm_cpu, output_cpu);
  b_ij_list = b_full.contiguous();

  grad_original = (grad_cpu * scaling_factors.view({1,1,1,-1})).contiguous();

}
