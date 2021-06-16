
conda activate latplan
# nvidia-tensorflow requires libm compiled against glibc-2.29
libm=$HOME/.local-new/glibc-2.29/lib/libm.so.6
[ -f $libm ] && export LD_PRELOAD=$libm
