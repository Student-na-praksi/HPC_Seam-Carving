## Useful commands

1. cache size
lscpu | egrep "L1d cache|L2 cache|L3 cache"
L1d cache:                               768 KiB (24 instances)
L2 cache:                                12 MiB (24 instances)
L3 cache:                                128 MiB (8 instances)


## Plan of imporvements
- recalcualte energy just for 3x3 neightbouthood of removed pixels, not whole image
- create non optimal calculate_energy() and see the difference (no MCARO, just one for loop)