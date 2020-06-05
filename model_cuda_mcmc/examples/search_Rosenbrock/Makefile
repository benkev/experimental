#
# Compile and build mcmc_interf.so Python extension module
# in the srs/ directory. 
#

TASKDIR = $(PWD)
MODELDIR = $(TASKDIR)
export MODELDIR

MODELFILE := $(shell find $(TASKDIR) -name '*model*.cu')

ifeq ($(MODELFILE),)
  $(error No "*model*.cu" file(s) found in $(TASKDIR))
endif

MODELFILE := $(firstword $(MODELFILE))
export MODELFILE


all:
	cd $(CUDA_MCMC)/src && $(MAKE) purge \
			&& python2 setup_mcmc.py build_ext --inplace \
			&& $(MAKE) clean && $(MAKE) install && cd $(TASKDIR)
	@echo "  "
	@echo MODEL is included from $(MODELFILE)
	@echo "  "

clean:
	cd $(CUDA_MCMC)/src && $(MAKE) clean && cd $(TASKDIR)

purge:
	cd $(CUDA_MCMC)/src && $(MAKE) purge && cd $(TASKDIR)

