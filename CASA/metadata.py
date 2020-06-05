import numpy as np

msmd.open('/data/1130643840/034445/ms_vv_rho_c169-170_f08-14_t034500/' \
          '1130643840_20151104034445_vv_rho_c169-170_f08-14.ms')


nant = msmd.nantennas()
antids = msmd.antennaids()

antoffs = np.zeros((nant,2), dtype=np.float64)

for iant in xrange(nant):
    aid = antids[iant]
    aof = msmd.antennaoffset(aid)
    antoffs[iant,0] = aof['longitude offset']['value']
    antoffs[iant,1] = aof['latitude offset']['value']





