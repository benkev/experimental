    #include <curand_kernel.h>

    extern "C"
    {
    __global__ void sim_drift(curandState *global_state, float const v, 
			      float const V, float const a, float const z, 
			      float const Z, float const t, float const T, 
			      float const dt, float const intra_sv, float *out)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, 
	  position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        curandState local_state = global_state[idx];

        /* Sample variability parameters. */
        start_delay = curand_uniform(&local_state)*T + (t-T/2);
        start_point = (curand_uniform(&local_state)*Z + (z-Z/2))*a;
        drift_rate = curand_normal(&local_state)*V + v;

        /* Set up drift variables. */
        prob_up = .5f*(1+sqrtf(dt)/intra_sv*drift_rate);
        step_size = sqrtf(dt)*intra_sv;
        time = start_delay;
        position = start_point;

        /* Simulate particle movement until threshold is crossed. */
        while (position > 0 & position < a) {
            rand = curand_uniform(&local_state);
            position += ((rand < prob_up)*2 - 1) * step_size;
            time += dt;
        }

        /* Save back state. */
        global_state[idx] = local_state;

        /* Figure out boundary, save result. */
        if (position <= 0) {
            out[idx] = -time;
        }
        else {
            out[idx] = time;
        }
    }

    __global__ void sim_drift_var_thresh(curandState *global_state, 
		        float const v, float const V, float const *a, 
                        float const z, float const Z, float const t, 
                        float const T, float const dt, float const intra_sv, 
                        int const a_len, float *out)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, 
	  position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int x_pos = 0;

        curandState local_state = global_state[idx];

        start_delay = curand_uniform(&local_state)*T + (t-T/2);
        start_point = curand_uniform(&local_state)*Z + (z-Z/2);
        drift_rate = curand_normal(&local_state)*V + v;

        prob_up = .5f*(1+sqrtf(dt)/intra_sv*drift_rate);
        step_size = sqrtf(dt)*intra_sv;
        time = 0;
        position = start_point;

        while (fabs(position) < a[x_pos] & time < a_len) {
            rand = curand_uniform(&local_state);
            position += ((rand < prob_up)*2 - 1) * step_size;
            time += dt;
            x_pos++;
        }

        time += start_delay;

        global_state[idx] = local_state;

        if (position <= 0) {
            out[idx] = -time;
        }
        else {
            out[idx] = time;
        }
    }

    __global__ void fill_normal(curandState *global_state, float *out)
    {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        curandState local_state = global_state[idx];

        out[idx] = curand_normal(&local_state);

        global_state[idx] = local_state;
    }

}







































/*    __global__ void sim_drift_switch(curandState *global_state, 
                        float const vpp, float const vcc, float const a, 
                        float const z, float const t, float const tcc, 
                        float const dt, float const intra_sv, float *out)
    {
        float start_delay, start_point, rand, prob_up_pp, prob_up_cc, 
	  position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        curandState local_state = global_state[idx];

        start_delay = t;
        start_point = z;

        prob_up_pp = .5f*(1+sqrtf(dt)/intra_sv*vpp);
        prob_up_cc = .5f*(1+sqrtf(dt)/intra_sv*vcc);

        step_size = sqrtf(dt)*intra_sv;
        time = 0;
        position = start_point;

        while (fabs(position) < a) {
            rand = curand_uniform(&local_state);
            if time < tcc {
                position += ((rand < prob_up_pp)*2 - 1) * step_size;
            }
            else {
                position += ((rand < prob_up_cc)*2 - 1) * step_size;
            time += dt;
        }

        time += start_delay;

        global_state[idx] = local_state;

        if (position <= 0) {
            out[idx] = -time;
        }
        else {
            out[idx] = time;
        }
    }
*/

 
