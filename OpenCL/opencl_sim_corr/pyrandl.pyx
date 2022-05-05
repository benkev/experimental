""" 
Module pyrandl contains class PyRandl. It provides random number generator
xoshiro256** 1.0 written in Cython for speed. 
The algorithms are obtained from the webpage by Prof. Sebastiano Vigna
"Xoshiro / xoroshiro generators and the PRNG shootout" at
http://xoshiro.di.unimi.it/

Compilation:
import pyximport; pyximport.install()
import pyrandl

"""

import numpy as np
cimport numpy as np

from cython cimport boundscheck, wraparound

# cython: boundscheck=False
# cython: wraparound=False


cdef class PyRandl(object):
	"""
	A small library of random number generators written in Cython for speed
	"""
	cdef unsigned long xstate, seed
	cdef int nstates
	cdef unsigned long c_0x9e3779b97f4a7c15
	cdef unsigned long c_0xbf58476d1ce4e5b9
	cdef unsigned long c_0x94d049bb133111eb
	
	
	cdef unsigned long JUMP[4]
	cdef unsigned long s0, s1, s2, s3
	
	cdef unsigned long [:,::1] rndst  # C-contiguous array of states
	cdef double [:] z1		  # Second normal from Box-Muller code
	cdef unsigned int [:] generate	  # Flags for Box-Muller code

	cdef double inv_ULONG_MAX
	cdef double two_pi
	
	# cdef extern from "vanvleck_structs.h":
	#	rand64State* rndst = (rand64State*) malloc(
	
	
	def __init__(self, seed=None, nstates=None):
		"""
		Class providing random number generators based on xoshiro256**.

		Instantiation:
		prng = PyRandl(seed=None, nstates=None)
		
			seed: any number of dtype=np.uint64. A single seed creates nstates
				  generators, the period of each being 2^256 - 1.

			nstates: total number of independent generators random numbers.
				  The 256-bit (i.e. four-int64-word) states of the nstates 
				  generators are generated in the rndst[nstates,4] array
				  and can be obtained with the method prng.get_rndst().
				  
		"""
		if seed <> None:
			self.seed = np.uint64(seed)
		else:
			self.seed = np.uint64(1234)	 # An arbitrary number
			
		self.xstate = self.seed	  # The xstate attribute is actually static

		self.c_0x9e3779b97f4a7c15 = np.uint64(0x9e3779b97f4a7c15)
		self.c_0xbf58476d1ce4e5b9 = np.uint64(0xbf58476d1ce4e5b9) 
		self.c_0x94d049bb133111eb = np.uint64(0x94d049bb133111eb)

		#self.c_0x180ec6d33cfd0aba = np.uint64(0x180ec6d33cfd0aba)
		

		self.JUMP[:] = [np.uint64(0x180ec6d33cfd0aba), \
						np.uint64(0xd5a61266f0c9392c), \
						np.uint64(0xa9582618e03fc9aa), \
						np.uint64(0x39abdc4529b1661c)]
		
		if nstates is None:
			self.nstates = 1
		else:
			self.nstates = nstates
			
		self.rndst = np.zeros((self.nstates,4), dtype=np.uint64)
		self.z1 = np.zeros((self.nstates), dtype=np.float64)
		self.generate = np.zeros((self.nstates), dtype=np.uint32)

		self.rand64_init()

		self.inv_ULONG_MAX = np.float64(1.)/np.float64(np.uint64(2**64-1))
		self.two_pi = 2. * np.pi
			

		
	cpdef unsigned long splitmix64(self):
		"""
		Splitmix64 is a PRNG with 64 bits of state and the period 2^64.
		Here it is used for "random" assigning of the initial multi-word states
		of other random number generators.
		Written in 2015 by Sebastiano Vigna (vigna@acm.org) 
		"""
		
		self.xstate += self.c_0x9e3779b97f4a7c15
		cdef unsigned long z = self.xstate
		z = (z ^ (z >> 30)) * self.c_0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * self.c_0x94d049bb133111eb
		return z ^ (z >> 31)



	
	cdef jump_ahead(self, int ist):
		"""
		This is the jump function for the xoshiro256** 1.0. generator. 
		It is equivalent to 2^128 calls to rand64_bits(); it can be used to 
		generate 2^128	non-overlapping subsequences for parallel computations.
		Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
		"""
		# static const cl_ulong JUMP[] = { 0x180ec6d33cfd0aba,
		#								 0xd5a61266f0c9392c, \
		#								 0xa9582618e03fc9aa, \
		#								 0x39abdc4529b1661c };
		cdef int i
		cdef unsigned long b
		cdef unsigned long s0 = 0
		cdef unsigned long s1 = 0
		cdef unsigned long s2 = 0
		cdef unsigned long s3 = 0

		for i in range(4):
			for b in range(64):
				if np.uint64(self.JUMP[i]) & np.uint64(1) << np.uint64(b):
					s0 ^= self.rndst[ist,0]
					s1 ^= self.rndst[ist,1]
					s2 ^= self.rndst[ist,2]
					s3 ^= self.rndst[ist,3]

				self.rand64_bits(ist)

		self.rndst[ist,0] = s0
		self.rndst[ist,1] = s1
		self.rndst[ist,2] = s2
		self.rndst[ist,3] = s3





	cpdef rand64_init(self):
		"""
		The rand64_init() function sets up initial 256-bit states using the
		given seed in an array rndst[4,nstates] for the xoshiro256** 1.0 
		generator (here rand64_bits() or randu64()). 

		Different seeds are guaranteed to produce different starting states and
		different sequences. The same seed always produces the same state and
		the same sequence.

		Sequences generated with different seeds usually do not have
		statistically correlated values, but some choices of seeds may give
		statistically correlated sequences. Sequences generated with the same
		seed and different sequence numbers will not have statistically
		correlated values.

		For the highest quality parallel pseudorandom number generation, each
		experiment should be assigned a unique seed. Within an experiment, each
		thread of computation should be assigned a unique sequence number. If an
		experiment spans multiple kernel launches, it is recommended that
		threads	between kernel launches be given the same seed, and sequence
		numbers be assigned in a monotonically increasing way. If the same
		configuration of threads is launched, random state can be preserved in
		global memory between launches to avoid state setup time.

		The text above has been plagiarized from the cuRAND documentation :)

		"The state must be seeded so that it is not everywhere zero. If you have
		a 64-bit seed, we suggest to seed a splitmix64 generator and use its
		output to fill s." (David Blackman and Sebastiano Vigna (vigna@acm.org))
		-- So we do here.
		"""

		cdef int ist

		self.rndst[0,0] = self.seed;
		self.rndst[0,1] = self.splitmix64()
		self.rndst[0,2] = self.splitmix64();
		self.rndst[0,3] = self.splitmix64();

		if self.nstates <= 1: return; # ============================== >>>

		for ist in range(1,self.nstates):
			self.rndst[ist,:] = self.rndst[ist-1,:]
			self.jump_ahead(ist)	


			



	cpdef unsigned long rand64_bits(self, int ist):
		"""
		call: rand64_bits(ist)

		Generate a single random number of dtype=np.uint64, in the stream ist.
		The numbers generated have uniform distribution between 0 and 2^64 - 1.
		The number of streams is specified at the class PyRandl instantiation.
		The period of each stream: 2^256 - 1.

		This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
		excellent (sub-ns) speed, a state (256 bits) that is large enough for
		any parallel application, and it passes all tests we are aware of.

		Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
		"""
		# cl_ulong *s = state->s; /* Just for brevity */
		
		cdef unsigned long [:] s

		cdef unsigned long result_starstar
		cdef unsigned long t

		s = self.rndst[ist,:]
		
		result_starstar = rotl(s[1] * 5, 7) * 9
		t = s[1] << 17

		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3], 45);

		return result_starstar;




	cpdef double randu64(self, int ist):
		"""
		call: randu64(ist)

		Generate a single random number of dtype=np.float, in the stream ist.
		The numbers generated have uniform distribution between 0 and 1.
		The number of streams is specified at the class PyRandl instantiation.
		The period of each stream: 2^256 - 1.

		This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
		excellent (sub-ns) speed, a state (256 bits) that is large enough for
		any parallel application, and it passes all tests we are aware of.

		Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
		"""
		# cl_ulong *s = state->s; /* Just for brevity */
		
		cdef unsigned long [:] s = self.rndst[ist,:]
		cdef unsigned long result_starstar
		cdef unsigned long t
		
		result_starstar = rotl(s[1] * 5, 7) * 9
		t = s[1] << 17
		
		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3], 45);
		
		#return np.float64(result_starstar) * self.inv_ULONG_MAX
		return result_starstar * self.inv_ULONG_MAX



	cpdef double randn64(self, int ist):
		"""
		Generate a single normally distributed random number
		using the Box-Muller transform.
		"""
		cdef double u1, u2, z0
		
		self.generate[ist] = 1 - self.generate[ist] # Toggle flag
		
		if self.generate[ist]:
			u1 = self.randu64(ist)
			u2 = self.randu64(ist)
			z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(self.two_pi * u2);
			self.z1[ist] = np.sqrt(-2.0 * np.log(u1)) * \
						   np.sin(self.two_pi * u2);
			return z0
		else:
			return self.z1[ist]

	

	cpdef get_rndst(self):
		"""
		Returns the array of random states rndst[nstates,4] created at the class
		instantiation.
		"""
		return np.asarray(self.rndst)
	

	
	cpdef unsigned long get_xstate(self):
		"""
		Returns the 64-bit state of splitmix64 generator.
		"""
		return self.xstate

	

	cpdef set_xstate(self, unsigned long xstate):
		"""
		Sets the 64-bit state of splitmix64 generator.
		"""
		self.xstate = xstate


		
	cpdef set_rndst(self, unsigned long [:,::1] rndst):
		"""
		Sets the xoshiro256** state rndst[4,nstates] from the argument	
		"""
		self.rndst = rndst
		self.nstates = rndst.shape[0]


		

cdef inline unsigned long rotl(unsigned long x, int k):
	""" Rotate left the 64-bit word x by k bits """
	return (x << k) | (x >> (64 - k))

