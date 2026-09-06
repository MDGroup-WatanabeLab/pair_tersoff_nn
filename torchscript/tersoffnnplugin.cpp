#include "lammpsplugin.h"
#include "version.h"

#include "pair_tersoff_nn.h"
#include <torch/script.h>

using namespace LAMMPS_NS;

static Pair *tersoff_nn_creator(LAMMPS *lmp)
{
  return new PairTersoffNN(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style   = "pair";
  plugin.name    = "tersoff_nn";
  plugin.info    = "Tersoff NN pair style using TorchScript";
  plugin.author  = "Yusuke Nishimura (yusukeskelton@toki.waseda.jp), Kotaro Takematsu (kotakematsu@toki.waseda.jp)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &tersoff_nn_creator;
  plugin.handle  = handle;
  (*register_plugin)(&plugin, lmp);
}
