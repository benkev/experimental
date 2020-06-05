__kernel void matrixTranspose(const int heightA,
							                const int widthA,
							                __global const float *a,
							                __global float *a_T) {
	const int alpha = 16;
	const int colA = get_global_id(0) * alpha;
	const int rowA = get_global_id(1) * alpha;

	if ((rowA < heightA) && (colA < widthA)) {

    __global float *src_vec1x16_0  = a + (rowA+ 0) * widthA + (colA+0);
    __global float *src_vec1x16_1  = a + (rowA+ 1) * widthA + (colA+0);
    __global float *src_vec1x16_2  = a + (rowA+ 2) * widthA + (colA+0);
    __global float *src_vec1x16_3  = a + (rowA+ 3) * widthA + (colA+0);
    __global float *src_vec1x16_4  = a + (rowA+ 4) * widthA + (colA+0);
    __global float *src_vec1x16_5  = a + (rowA+ 5) * widthA + (colA+0);
    __global float *src_vec1x16_6  = a + (rowA+ 6) * widthA + (colA+0);
    __global float *src_vec1x16_7  = a + (rowA+ 7) * widthA + (colA+0);
    __global float *src_vec1x16_8  = a + (rowA+ 8) * widthA + (colA+0);
    __global float *src_vec1x16_9  = a + (rowA+ 9) * widthA + (colA+0);
    __global float *src_vec1x16_10 = a + (rowA+10) * widthA + (colA+0);
    __global float *src_vec1x16_11 = a + (rowA+11) * widthA + (colA+0);
    __global float *src_vec1x16_12 = a + (rowA+12) * widthA + (colA+0);
    __global float *src_vec1x16_13 = a + (rowA+13) * widthA + (colA+0);
    __global float *src_vec1x16_14 = a + (rowA+14) * widthA + (colA+0);
    __global float *src_vec1x16_15 = a + (rowA+15) * widthA + (colA+0);

    float16 vec1x16_0  = *((__global float16 *)src_vec1x16_0);
    float16 vec1x16_1  = *((__global float16 *)src_vec1x16_1);
    float16 vec1x16_2  = *((__global float16 *)src_vec1x16_2);
    float16 vec1x16_3  = *((__global float16 *)src_vec1x16_3);
    float16 vec1x16_4  = *((__global float16 *)src_vec1x16_4);
    float16 vec1x16_5  = *((__global float16 *)src_vec1x16_5);
    float16 vec1x16_6  = *((__global float16 *)src_vec1x16_6);
    float16 vec1x16_7  = *((__global float16 *)src_vec1x16_7);
    float16 vec1x16_8  = *((__global float16 *)src_vec1x16_8);
    float16 vec1x16_9  = *((__global float16 *)src_vec1x16_9);
    float16 vec1x16_10 = *((__global float16 *)src_vec1x16_10);
    float16 vec1x16_11 = *((__global float16 *)src_vec1x16_11);
    float16 vec1x16_12 = *((__global float16 *)src_vec1x16_12);
    float16 vec1x16_13 = *((__global float16 *)src_vec1x16_13);
    float16 vec1x16_14 = *((__global float16 *)src_vec1x16_14);
    float16 vec1x16_15 = *((__global float16 *)src_vec1x16_15);
     
		a_T[(colA+ 0) * heightA + (rowA+0)] = vec1x16_0.s0;
		a_T[(colA+ 1) * heightA + (rowA+0)] = vec1x16_0.s1;
   	a_T[(colA+ 2) * heightA + (rowA+0)] = vec1x16_0.s2;
    a_T[(colA+ 3) * heightA + (rowA+0)] = vec1x16_0.s3;
		a_T[(colA+ 4) * heightA + (rowA+0)] = vec1x16_0.s4;
		a_T[(colA+ 5) * heightA + (rowA+0)] = vec1x16_0.s5;
   	a_T[(colA+ 6) * heightA + (rowA+0)] = vec1x16_0.s6;
    a_T[(colA+ 7) * heightA + (rowA+0)] = vec1x16_0.s7;
		a_T[(colA+ 8) * heightA + (rowA+0)] = vec1x16_0.s8;
		a_T[(colA+ 9) * heightA + (rowA+0)] = vec1x16_0.s9;
   	a_T[(colA+10) * heightA + (rowA+0)] = vec1x16_0.sa;
    a_T[(colA+11) * heightA + (rowA+0)] = vec1x16_0.sb;
		a_T[(colA+12) * heightA + (rowA+0)] = vec1x16_0.sc;
		a_T[(colA+13) * heightA + (rowA+0)] = vec1x16_0.sd;
   	a_T[(colA+14) * heightA + (rowA+0)] = vec1x16_0.se;
    a_T[(colA+15) * heightA + (rowA+0)] = vec1x16_0.sf;

		a_T[(colA+ 0) * heightA + (rowA+1)] = vec1x16_1.s0;
		a_T[(colA+ 1) * heightA + (rowA+1)] = vec1x16_1.s1;
   	a_T[(colA+ 2) * heightA + (rowA+1)] = vec1x16_1.s2;
    a_T[(colA+ 3) * heightA + (rowA+1)] = vec1x16_1.s3;
		a_T[(colA+ 4) * heightA + (rowA+1)] = vec1x16_1.s4;
		a_T[(colA+ 5) * heightA + (rowA+1)] = vec1x16_1.s5;
   	a_T[(colA+ 6) * heightA + (rowA+1)] = vec1x16_1.s6;
    a_T[(colA+ 7) * heightA + (rowA+1)] = vec1x16_1.s7;
		a_T[(colA+ 8) * heightA + (rowA+1)] = vec1x16_1.s8;
		a_T[(colA+ 9) * heightA + (rowA+1)] = vec1x16_1.s9;
   	a_T[(colA+10) * heightA + (rowA+1)] = vec1x16_1.sa;
    a_T[(colA+11) * heightA + (rowA+1)] = vec1x16_1.sb;
		a_T[(colA+12) * heightA + (rowA+1)] = vec1x16_1.sc;
		a_T[(colA+13) * heightA + (rowA+1)] = vec1x16_1.sd;
   	a_T[(colA+14) * heightA + (rowA+1)] = vec1x16_1.se;
    a_T[(colA+15) * heightA + (rowA+1)] = vec1x16_1.sf;

		a_T[(colA+ 0) * heightA + (rowA+2)] = vec1x16_2.s0;
		a_T[(colA+ 1) * heightA + (rowA+2)] = vec1x16_2.s1;
   	a_T[(colA+ 2) * heightA + (rowA+2)] = vec1x16_2.s2;
    a_T[(colA+ 3) * heightA + (rowA+2)] = vec1x16_2.s3;
		a_T[(colA+ 4) * heightA + (rowA+2)] = vec1x16_2.s4;
		a_T[(colA+ 5) * heightA + (rowA+2)] = vec1x16_2.s5;
   	a_T[(colA+ 6) * heightA + (rowA+2)] = vec1x16_2.s6;
    a_T[(colA+ 7) * heightA + (rowA+2)] = vec1x16_2.s7;
		a_T[(colA+ 8) * heightA + (rowA+2)] = vec1x16_2.s8;
		a_T[(colA+ 9) * heightA + (rowA+2)] = vec1x16_2.s9;
   	a_T[(colA+10) * heightA + (rowA+2)] = vec1x16_2.sa;
    a_T[(colA+11) * heightA + (rowA+2)] = vec1x16_2.sb;
		a_T[(colA+12) * heightA + (rowA+2)] = vec1x16_2.sc;
		a_T[(colA+13) * heightA + (rowA+2)] = vec1x16_2.sd;
   	a_T[(colA+14) * heightA + (rowA+2)] = vec1x16_2.se;
    a_T[(colA+15) * heightA + (rowA+2)] = vec1x16_2.sf;

		a_T[(colA+ 0) * heightA + (rowA+3)] = vec1x16_3.s0;
		a_T[(colA+ 1) * heightA + (rowA+3)] = vec1x16_3.s1;
   	a_T[(colA+ 2) * heightA + (rowA+3)] = vec1x16_3.s2;
    a_T[(colA+ 3) * heightA + (rowA+3)] = vec1x16_3.s3;
		a_T[(colA+ 4) * heightA + (rowA+3)] = vec1x16_3.s4;
		a_T[(colA+ 5) * heightA + (rowA+3)] = vec1x16_3.s5;
   	a_T[(colA+ 6) * heightA + (rowA+3)] = vec1x16_3.s6;
    a_T[(colA+ 7) * heightA + (rowA+3)] = vec1x16_3.s7;
		a_T[(colA+ 8) * heightA + (rowA+3)] = vec1x16_3.s8;
		a_T[(colA+ 9) * heightA + (rowA+3)] = vec1x16_3.s9;
   	a_T[(colA+10) * heightA + (rowA+3)] = vec1x16_3.sa;
    a_T[(colA+11) * heightA + (rowA+3)] = vec1x16_3.sb;
		a_T[(colA+12) * heightA + (rowA+3)] = vec1x16_3.sc;
		a_T[(colA+13) * heightA + (rowA+3)] = vec1x16_3.sd;
   	a_T[(colA+14) * heightA + (rowA+3)] = vec1x16_3.se;
    a_T[(colA+15) * heightA + (rowA+3)] = vec1x16_3.sf;

		a_T[(colA+ 0) * heightA + (rowA+4)] = vec1x16_4.s0;
		a_T[(colA+ 1) * heightA + (rowA+4)] = vec1x16_4.s1;
   	a_T[(colA+ 2) * heightA + (rowA+4)] = vec1x16_4.s2;
    a_T[(colA+ 3) * heightA + (rowA+4)] = vec1x16_4.s3;
		a_T[(colA+ 4) * heightA + (rowA+4)] = vec1x16_4.s4;
		a_T[(colA+ 5) * heightA + (rowA+4)] = vec1x16_4.s5;
   	a_T[(colA+ 6) * heightA + (rowA+4)] = vec1x16_4.s6;
    a_T[(colA+ 7) * heightA + (rowA+4)] = vec1x16_4.s7;
		a_T[(colA+ 8) * heightA + (rowA+4)] = vec1x16_4.s8;
		a_T[(colA+ 9) * heightA + (rowA+4)] = vec1x16_4.s9;
   	a_T[(colA+10) * heightA + (rowA+4)] = vec1x16_4.sa;
    a_T[(colA+11) * heightA + (rowA+4)] = vec1x16_4.sb;
		a_T[(colA+12) * heightA + (rowA+4)] = vec1x16_4.sc;
		a_T[(colA+13) * heightA + (rowA+4)] = vec1x16_4.sd;
   	a_T[(colA+14) * heightA + (rowA+4)] = vec1x16_4.se;
    a_T[(colA+15) * heightA + (rowA+4)] = vec1x16_4.sf;

		a_T[(colA+ 0) * heightA + (rowA+5)] = vec1x16_5.s0;
		a_T[(colA+ 1) * heightA + (rowA+5)] = vec1x16_5.s1;
   	a_T[(colA+ 2) * heightA + (rowA+5)] = vec1x16_5.s2;
    a_T[(colA+ 3) * heightA + (rowA+5)] = vec1x16_5.s3;
		a_T[(colA+ 4) * heightA + (rowA+5)] = vec1x16_5.s4;
		a_T[(colA+ 5) * heightA + (rowA+5)] = vec1x16_5.s5;
   	a_T[(colA+ 6) * heightA + (rowA+5)] = vec1x16_5.s6;
    a_T[(colA+ 7) * heightA + (rowA+5)] = vec1x16_5.s7;
		a_T[(colA+ 8) * heightA + (rowA+5)] = vec1x16_5.s8;
		a_T[(colA+ 9) * heightA + (rowA+5)] = vec1x16_5.s9;
   	a_T[(colA+10) * heightA + (rowA+5)] = vec1x16_5.sa;
    a_T[(colA+11) * heightA + (rowA+5)] = vec1x16_5.sb;
		a_T[(colA+12) * heightA + (rowA+5)] = vec1x16_5.sc;
		a_T[(colA+13) * heightA + (rowA+5)] = vec1x16_5.sd;
   	a_T[(colA+14) * heightA + (rowA+5)] = vec1x16_5.se;
    a_T[(colA+15) * heightA + (rowA+5)] = vec1x16_5.sf;

		a_T[(colA+ 0) * heightA + (rowA+6)] = vec1x16_6.s0;
		a_T[(colA+ 1) * heightA + (rowA+6)] = vec1x16_6.s1;
   	a_T[(colA+ 2) * heightA + (rowA+6)] = vec1x16_6.s2;
    a_T[(colA+ 3) * heightA + (rowA+6)] = vec1x16_6.s3;
		a_T[(colA+ 4) * heightA + (rowA+6)] = vec1x16_6.s4;
		a_T[(colA+ 5) * heightA + (rowA+6)] = vec1x16_6.s5;
   	a_T[(colA+ 6) * heightA + (rowA+6)] = vec1x16_6.s6;
    a_T[(colA+ 7) * heightA + (rowA+6)] = vec1x16_6.s7;
		a_T[(colA+ 8) * heightA + (rowA+6)] = vec1x16_6.s8;
		a_T[(colA+ 9) * heightA + (rowA+6)] = vec1x16_6.s9;
   	a_T[(colA+10) * heightA + (rowA+6)] = vec1x16_6.sa;
    a_T[(colA+11) * heightA + (rowA+6)] = vec1x16_6.sb;
		a_T[(colA+12) * heightA + (rowA+6)] = vec1x16_6.sc;
		a_T[(colA+13) * heightA + (rowA+6)] = vec1x16_6.sd;
   	a_T[(colA+14) * heightA + (rowA+6)] = vec1x16_6.se;
    a_T[(colA+15) * heightA + (rowA+6)] = vec1x16_6.sf;

		a_T[(colA+ 0) * heightA + (rowA+7)] = vec1x16_7.s0;
		a_T[(colA+ 1) * heightA + (rowA+7)] = vec1x16_7.s1;
   	a_T[(colA+ 2) * heightA + (rowA+7)] = vec1x16_7.s2;
    a_T[(colA+ 3) * heightA + (rowA+7)] = vec1x16_7.s3;
		a_T[(colA+ 4) * heightA + (rowA+7)] = vec1x16_7.s4;
		a_T[(colA+ 5) * heightA + (rowA+7)] = vec1x16_7.s5;
   	a_T[(colA+ 6) * heightA + (rowA+7)] = vec1x16_7.s6;
    a_T[(colA+ 7) * heightA + (rowA+7)] = vec1x16_7.s7;
		a_T[(colA+ 8) * heightA + (rowA+7)] = vec1x16_7.s8;
		a_T[(colA+ 9) * heightA + (rowA+7)] = vec1x16_7.s9;
   	a_T[(colA+10) * heightA + (rowA+7)] = vec1x16_7.sa;
    a_T[(colA+11) * heightA + (rowA+7)] = vec1x16_7.sb;
		a_T[(colA+12) * heightA + (rowA+7)] = vec1x16_7.sc;
		a_T[(colA+13) * heightA + (rowA+7)] = vec1x16_7.sd;
   	a_T[(colA+14) * heightA + (rowA+7)] = vec1x16_7.se;
    a_T[(colA+15) * heightA + (rowA+7)] = vec1x16_7.sf;

		a_T[(colA+ 0) * heightA + (rowA+8)] = vec1x16_8.s0;
		a_T[(colA+ 1) * heightA + (rowA+8)] = vec1x16_8.s1;
   	a_T[(colA+ 2) * heightA + (rowA+8)] = vec1x16_8.s2;
    a_T[(colA+ 3) * heightA + (rowA+8)] = vec1x16_8.s3;
		a_T[(colA+ 4) * heightA + (rowA+8)] = vec1x16_8.s4;
		a_T[(colA+ 5) * heightA + (rowA+8)] = vec1x16_8.s5;
   	a_T[(colA+ 6) * heightA + (rowA+8)] = vec1x16_8.s6;
    a_T[(colA+ 7) * heightA + (rowA+8)] = vec1x16_8.s7;
		a_T[(colA+ 8) * heightA + (rowA+8)] = vec1x16_8.s8;
		a_T[(colA+ 9) * heightA + (rowA+8)] = vec1x16_8.s9;
   	a_T[(colA+10) * heightA + (rowA+8)] = vec1x16_8.sa;
    a_T[(colA+11) * heightA + (rowA+8)] = vec1x16_8.sb;
		a_T[(colA+12) * heightA + (rowA+8)] = vec1x16_8.sc;
		a_T[(colA+13) * heightA + (rowA+8)] = vec1x16_8.sd;
   	a_T[(colA+14) * heightA + (rowA+8)] = vec1x16_8.se;
    a_T[(colA+15) * heightA + (rowA+8)] = vec1x16_8.sf;

		a_T[(colA+ 0) * heightA + (rowA+9)] = vec1x16_9.s0;
		a_T[(colA+ 1) * heightA + (rowA+9)] = vec1x16_9.s1;
   	a_T[(colA+ 2) * heightA + (rowA+9)] = vec1x16_9.s2;
    a_T[(colA+ 3) * heightA + (rowA+9)] = vec1x16_9.s3;
		a_T[(colA+ 4) * heightA + (rowA+9)] = vec1x16_9.s4;
		a_T[(colA+ 5) * heightA + (rowA+9)] = vec1x16_9.s5;
   	a_T[(colA+ 6) * heightA + (rowA+9)] = vec1x16_9.s6;
    a_T[(colA+ 7) * heightA + (rowA+9)] = vec1x16_9.s7;
		a_T[(colA+ 8) * heightA + (rowA+9)] = vec1x16_9.s8;
		a_T[(colA+ 9) * heightA + (rowA+9)] = vec1x16_9.s9;
   	a_T[(colA+10) * heightA + (rowA+9)] = vec1x16_9.sa;
    a_T[(colA+11) * heightA + (rowA+9)] = vec1x16_9.sb;
		a_T[(colA+12) * heightA + (rowA+9)] = vec1x16_9.sc;
		a_T[(colA+13) * heightA + (rowA+9)] = vec1x16_9.sd;
   	a_T[(colA+14) * heightA + (rowA+9)] = vec1x16_9.se;
    a_T[(colA+15) * heightA + (rowA+9)] = vec1x16_9.sf;

		a_T[(colA+ 0) * heightA + (rowA+10)] = vec1x16_10.s0;
		a_T[(colA+ 1) * heightA + (rowA+10)] = vec1x16_10.s1;
   	a_T[(colA+ 2) * heightA + (rowA+10)] = vec1x16_10.s2;
    a_T[(colA+ 3) * heightA + (rowA+10)] = vec1x16_10.s3;
		a_T[(colA+ 4) * heightA + (rowA+10)] = vec1x16_10.s4;
		a_T[(colA+ 5) * heightA + (rowA+10)] = vec1x16_10.s5;
   	a_T[(colA+ 6) * heightA + (rowA+10)] = vec1x16_10.s6;
    a_T[(colA+ 7) * heightA + (rowA+10)] = vec1x16_10.s7;
		a_T[(colA+ 8) * heightA + (rowA+10)] = vec1x16_10.s8;
		a_T[(colA+ 9) * heightA + (rowA+10)] = vec1x16_10.s9;
   	a_T[(colA+10) * heightA + (rowA+10)] = vec1x16_10.sa;
    a_T[(colA+11) * heightA + (rowA+10)] = vec1x16_10.sb;
		a_T[(colA+12) * heightA + (rowA+10)] = vec1x16_10.sc;
		a_T[(colA+13) * heightA + (rowA+10)] = vec1x16_10.sd;
   	a_T[(colA+14) * heightA + (rowA+10)] = vec1x16_10.se;
    a_T[(colA+15) * heightA + (rowA+10)] = vec1x16_10.sf;

		a_T[(colA+ 0) * heightA + (rowA+11)] = vec1x16_11.s0;
		a_T[(colA+ 1) * heightA + (rowA+11)] = vec1x16_11.s1;
   	a_T[(colA+ 2) * heightA + (rowA+11)] = vec1x16_11.s2;
    a_T[(colA+ 3) * heightA + (rowA+11)] = vec1x16_11.s3;
		a_T[(colA+ 4) * heightA + (rowA+11)] = vec1x16_11.s4;
		a_T[(colA+ 5) * heightA + (rowA+11)] = vec1x16_11.s5;
   	a_T[(colA+ 6) * heightA + (rowA+11)] = vec1x16_11.s6;
    a_T[(colA+ 7) * heightA + (rowA+11)] = vec1x16_11.s7;
		a_T[(colA+ 8) * heightA + (rowA+11)] = vec1x16_11.s8;
		a_T[(colA+ 9) * heightA + (rowA+11)] = vec1x16_11.s9;
   	a_T[(colA+10) * heightA + (rowA+11)] = vec1x16_11.sa;
    a_T[(colA+11) * heightA + (rowA+11)] = vec1x16_11.sb;
		a_T[(colA+12) * heightA + (rowA+11)] = vec1x16_11.sc;
		a_T[(colA+13) * heightA + (rowA+11)] = vec1x16_11.sd;
   	a_T[(colA+14) * heightA + (rowA+11)] = vec1x16_11.se;
    a_T[(colA+15) * heightA + (rowA+11)] = vec1x16_11.sf;

		a_T[(colA+ 0) * heightA + (rowA+12)] = vec1x16_12.s0;
		a_T[(colA+ 1) * heightA + (rowA+12)] = vec1x16_12.s1;
   	a_T[(colA+ 2) * heightA + (rowA+12)] = vec1x16_12.s2;
    a_T[(colA+ 3) * heightA + (rowA+12)] = vec1x16_12.s3;
		a_T[(colA+ 4) * heightA + (rowA+12)] = vec1x16_12.s4;
		a_T[(colA+ 5) * heightA + (rowA+12)] = vec1x16_12.s5;
   	a_T[(colA+ 6) * heightA + (rowA+12)] = vec1x16_12.s6;
    a_T[(colA+ 7) * heightA + (rowA+12)] = vec1x16_12.s7;
		a_T[(colA+ 8) * heightA + (rowA+12)] = vec1x16_12.s8;
		a_T[(colA+ 9) * heightA + (rowA+12)] = vec1x16_12.s9;
   	a_T[(colA+10) * heightA + (rowA+12)] = vec1x16_12.sa;
    a_T[(colA+11) * heightA + (rowA+12)] = vec1x16_12.sb;
		a_T[(colA+12) * heightA + (rowA+12)] = vec1x16_12.sc;
		a_T[(colA+13) * heightA + (rowA+12)] = vec1x16_12.sd;
   	a_T[(colA+14) * heightA + (rowA+12)] = vec1x16_12.se;
    a_T[(colA+15) * heightA + (rowA+12)] = vec1x16_12.sf;

		a_T[(colA+ 0) * heightA + (rowA+13)] = vec1x16_13.s0;
		a_T[(colA+ 1) * heightA + (rowA+13)] = vec1x16_13.s1;
   	a_T[(colA+ 2) * heightA + (rowA+13)] = vec1x16_13.s2;
    a_T[(colA+ 3) * heightA + (rowA+13)] = vec1x16_13.s3;
		a_T[(colA+ 4) * heightA + (rowA+13)] = vec1x16_13.s4;
		a_T[(colA+ 5) * heightA + (rowA+13)] = vec1x16_13.s5;
   	a_T[(colA+ 6) * heightA + (rowA+13)] = vec1x16_13.s6;
    a_T[(colA+ 7) * heightA + (rowA+13)] = vec1x16_13.s7;
		a_T[(colA+ 8) * heightA + (rowA+13)] = vec1x16_13.s8;
		a_T[(colA+ 9) * heightA + (rowA+13)] = vec1x16_13.s9;
   	a_T[(colA+10) * heightA + (rowA+13)] = vec1x16_13.sa;
    a_T[(colA+11) * heightA + (rowA+13)] = vec1x16_13.sb;
		a_T[(colA+12) * heightA + (rowA+13)] = vec1x16_13.sc;
		a_T[(colA+13) * heightA + (rowA+13)] = vec1x16_13.sd;
   	a_T[(colA+14) * heightA + (rowA+13)] = vec1x16_13.se;
    a_T[(colA+15) * heightA + (rowA+13)] = vec1x16_13.sf;

		a_T[(colA+ 0) * heightA + (rowA+14)] = vec1x16_14.s0;
		a_T[(colA+ 1) * heightA + (rowA+14)] = vec1x16_14.s1;
   	a_T[(colA+ 2) * heightA + (rowA+14)] = vec1x16_14.s2;
    a_T[(colA+ 3) * heightA + (rowA+14)] = vec1x16_14.s3;
		a_T[(colA+ 4) * heightA + (rowA+14)] = vec1x16_14.s4;
		a_T[(colA+ 5) * heightA + (rowA+14)] = vec1x16_14.s5;
   	a_T[(colA+ 6) * heightA + (rowA+14)] = vec1x16_14.s6;
    a_T[(colA+ 7) * heightA + (rowA+14)] = vec1x16_14.s7;
		a_T[(colA+ 8) * heightA + (rowA+14)] = vec1x16_14.s8;
		a_T[(colA+ 9) * heightA + (rowA+14)] = vec1x16_14.s9;
   	a_T[(colA+10) * heightA + (rowA+14)] = vec1x16_14.sa;
    a_T[(colA+11) * heightA + (rowA+14)] = vec1x16_14.sb;
		a_T[(colA+12) * heightA + (rowA+14)] = vec1x16_14.sc;
		a_T[(colA+13) * heightA + (rowA+14)] = vec1x16_14.sd;
   	a_T[(colA+14) * heightA + (rowA+14)] = vec1x16_14.se;
    a_T[(colA+15) * heightA + (rowA+14)] = vec1x16_14.sf;

		a_T[(colA+ 0) * heightA + (rowA+15)] = vec1x16_15.s0;
		a_T[(colA+ 1) * heightA + (rowA+15)] = vec1x16_15.s1;
   	a_T[(colA+ 2) * heightA + (rowA+15)] = vec1x16_15.s2;
    a_T[(colA+ 3) * heightA + (rowA+15)] = vec1x16_15.s3;
		a_T[(colA+ 4) * heightA + (rowA+15)] = vec1x16_15.s4;
		a_T[(colA+ 5) * heightA + (rowA+15)] = vec1x16_15.s5;
   	a_T[(colA+ 6) * heightA + (rowA+15)] = vec1x16_15.s6;
    a_T[(colA+ 7) * heightA + (rowA+15)] = vec1x16_15.s7;
		a_T[(colA+ 8) * heightA + (rowA+15)] = vec1x16_15.s8;
		a_T[(colA+ 9) * heightA + (rowA+15)] = vec1x16_15.s9;
   	a_T[(colA+10) * heightA + (rowA+15)] = vec1x16_15.sa;
    a_T[(colA+11) * heightA + (rowA+15)] = vec1x16_15.sb;
		a_T[(colA+12) * heightA + (rowA+15)] = vec1x16_15.sc;
		a_T[(colA+13) * heightA + (rowA+15)] = vec1x16_15.sd;
   	a_T[(colA+14) * heightA + (rowA+15)] = vec1x16_15.se;
    a_T[(colA+15) * heightA + (rowA+15)] = vec1x16_15.sf;

	}

}
