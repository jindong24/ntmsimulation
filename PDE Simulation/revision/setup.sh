# Source from Bash or Zsh after activating the FEniCS environment.
export NTM_REVISION="$(pwd -P)"
export NTM_RUNS="${NTM_RUNS:-$NTM_REVISION/runs}"
export NTM_ORIGINAL_MESH="$NTM_REVISION/reproduce/mesh/original_mshr.h5"
export PYTHONPATH="$NTM_REVISION/reproduce/solver/src:$NTM_REVISION/reproduce/solver/scripts${PYTHONPATH:+:$PYTHONPATH}"
case "$(uname -s)" in
  Darwin) export NTM_UFC_LIBRARY="${NTM_UFC_LIBRARY:-$NTM_REVISION/build/ufc_batch.dylib}" ;;
  *) export NTM_UFC_LIBRARY="${NTM_UFC_LIBRARY:-$NTM_REVISION/build/ufc_batch.so}" ;;
esac
export OMP_NUM_THREADS="${NTM_THREADS_PER_WORKER:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
