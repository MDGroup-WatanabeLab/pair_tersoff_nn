#include "lammpsplugin.h"
#include "version.h"

#include "pair_tersoff_nn_direct.h"

using namespace LAMMPS_NS;

static Pair *tersoff_nn_direct_creator(LAMMPS *lmp)
{
  return new PairTersoffNNDirect(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style   = "pair";
  plugin.name    = "tersoff_nn_direct";
  plugin.info    = "Tersoff NN pair style with hand-written MLP inference";
  plugin.author  = "Yusuke Nishimura";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &tersoff_nn_direct_creator;
  plugin.handle  = handle;
  (*register_plugin)(&plugin, lmp);
}
