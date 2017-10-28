/*
 * C++ Templete for a Binarized CNN
 *
 *  Created on: 2017/07/01
 *      Author: H. Nakahara
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>

#include <ap_int.h>

#ifdef __SDSCC__
#include "sds_lib.h"
#else 
#define sds_alloc(x)(malloc(x))
#define sds_free(x)(free(x))
#endif

// custom bitwidth for streaming operation
typedef ap_int<2>    bit_2;
typedef ap_int<4>    bit_4;
typedef ap_int<8>    bit_8;
typedef ap_int<16>   bit_16;
typedef ap_int<32>   bit_32;
typedef ap_int<64>   bit_64;
typedef ap_int<128>  bit_128;
typedef ap_int<256>  bit_256;
typedef ap_int<512>  bit_512;

// weight memory -----------------------------------------------------------
(DEF_WEIGHT_MEM)
// bias memory ------------------------------------------------------------
(DEF_BIAS_MEM)
// -------------------------------------------------------------------------
// Load weights and bias from the external memory (DDR3/4 Memory)
// -------------------------------------------------------------------------
#ifdef __SDSCC__
#pragma SDS data access_pattern(t_bin_convW: SEQUENTIAL)
#pragma SDS data access_pattern(t_BNFb: SEQUENTIAL)
#pragma SDS data zero_copy(t_bin_convW[0:(WEIGHT_SIZ)])
#pragma SDS data zero_copy(t_BNFb[0:(BIAS_SIZ)])
#endif
void setup(
#ifdef __SDSCC__
	    int *t_bin_convW,
		int *t_BNFb
#else 
        int t_bin_convW[(WEIGHT_SIZ)],
        int t_BNFb[(BIAS_SIZ)]
#endif
)
{
	// set buffer memory -----------------------------------------------
	int x, y, of, inf, offset;

	// -----------------------------------------------------------------
	// setup memory
	// -----------------------------------------------------------------
(SET_WEIGHT_MEM)
(SET_BIAS_MEM)
}

// -------------------------------------------------------------------------
// Binary Convolutional Layer
// -------------------------------------------------------------------------
void bin_conv2d_pipeline(
		ap_int<(MAX_BCONV_WIDTH)> fmap[(IMGSIZ)][(IMGSIZ)],
		int layer,
		int size,
		int n_in,
		int n_out
		)
{
(BCONV_REG_PRAGMA)

	int ofeat, infeat, w_flag;
	int i, k, ky, kx, ix, iy, ox, oy;
	int idx = 0;

	static ap_int<(MAX_BCONV_WIDTH)> shift_reg1[((IMGSIZ)+2)*3];
#pragma HLS ARRAY_PARTITION variable=shift_reg1 complete dim=1
	static ap_uint<1> padding_shift_reg[((IMGSIZ)+2)*3];
#pragma HLS ARRAY_PARTITION variable=padding_shift_reg complete dim=1

	int cnt = 0;

	ix = iy = ox = oy = w_flag = 0;

    CONV_IF: for( k = 0; k < (size+2) * (size+2); k++){
#pragma HLS loop_flatten off

    	SHIFT_REG: for( i = 0; i < 2 * ((IMGSIZ)+2) + 3; i++){
#pragma HLS UNROLL
    		shift_reg1[ i] = shift_reg1[ i + 1];
    		padding_shift_reg[ i] = padding_shift_reg[ i + 1];
    	}
    	ap_int<(MAX_BCONV_WIDTH)> din;
    	ap_uint<1> padding;
    	if( (ix > 0 && ix <= size) && (iy > 0 && iy <= size)){
		din = (ap_int<(MAX_BCONV_WIDTH)>)fmap[iy-1][ix-1];
		padding = 0;
    	} else {
    		ap_int<(MAX_BCONV_WIDTH)> allone;
    		allone = ~0;
    		din = allone;
		padding = 1;
    	}
    	switch( layer){
(BCONV_REG_SELECT)
    	}

    	ix++;
    	if( ix == size+2){
    		ix = 0;
    		iy++;
    	}

    	if( k >= ((size+2)*2+3 - 1)){
    		w_flag++;
    		if( w_flag > (size+2)){
            	w_flag = 1;
            	cnt    = 0;
            }
    	}

    	// convolutional operation -----------------------------------
		ap_uint<(MAX_BCONV_WIDTH)> bit_tmp = 0x1;
		ap_uint<(MAX_BCONV_WIDTH)> streamOut = 0;

    	OF: for( ofeat = 0; ofeat < n_out; ofeat++){
    		ap_int<16> tmp = 0;
    		ap_int<16> tmp2;

            CONV_KY: for( ky = 0; ky < 3; ky++){
#pragma HLS pipeline
            	CONV_KX: for( kx = 0; kx < 3; kx++){
            		ap_uint<(MAX_BCONV_WIDTH)> bx, bw;
            		ap_uint<(MAX_BCONV_WIDTH)> bxor;
                    ap_uint<(MAX_BCONV_WIDTH)> mask;
                    ap_uint<(MAX_BCONV_WIDTH)> allzero = 0;
                    ap_uint<1>is_padding;

            		switch( layer){
(BCONV_WEIGHT_SELECT)
            		}

                    (BIN_XOR_MAC)

			tmp2 = 0;
                    ONES_COUNT: for( i = 0; i < (MAX_BCONV_WIDTH); i++){
                        tmp2 += (((bxor >> i) & 0x1) == 1) ? 1 : 0;
                    }
                    if( is_padding == 0)
                        tmp += (n_in - tmp2 * 2);
		}
            }

            if( w_flag > 0 && w_flag <= size){
#pragma HLS pipeline
            	ap_int<16> bias;
            	switch( layer){
(BCONV_BIAS_SELECT)
            	}
            	tmp += bias;

            	if( tmp >= 0) streamOut = streamOut | bit_tmp;

            	bit_tmp = bit_tmp << 1;

            	cnt++;
            	if( cnt == n_out){
            		cnt = 0;
            		fmap[oy][ox] = (ap_int<(MAX_BCONV_WIDTH)>)streamOut;

            		ox++;
            		if( ox == size){
            			ox = 0;
            			oy++;
            		}

            		idx++;
            	}

            }
    	}

    }
}

// ------------------------------------------------------------------------
template< typename BIN_TYPE, typename BOUT_TYPE, int N_IFEAT, int N_OFEAT, int IF_SIZ, int OF_SIZ>
void int_conv2d_pipeline(
		BIN_TYPE infmap[IF_SIZ][IF_SIZ],
		BOUT_TYPE outfmap[OF_SIZ][OF_SIZ],
		ap_int<(NUMIMG)> W[N_OFEAT][3*3],
		ap_int<20> BNFb[N_OFEAT]
		)
{
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=9 dim=2

	int ofeat, infeat;
	int w_flag;
	int i, k, ky, kx;

	int idx = 0;

	static ap_int<N_IFEAT> shift_reg1[(IF_SIZ+2)*3];
#pragma HLS ARRAY_PARTITION variable=shift_reg1 complete dim=1
	int cnt = 0;

	int debug_out = 0;
    w_flag = 0;

    int ix, iy, ox, oy;
    ix = iy = ox = oy = 0;

    CONV_IF: for( k = 0; k < (IF_SIZ+2) * (IF_SIZ+2); k++){
#pragma HLS loop_flatten off

    	// pipeline register ------------------------------------------
    	SHIFT_REG: for( i = 0; i < 2 * (IF_SIZ+2) + 3; i++){
#pragma HLS UNROLL
    		shift_reg1[ i] = shift_reg1[ i + 1];
    	}
    	ap_int<N_IFEAT> din;
    	if( (ix > 0 && ix <= IF_SIZ) && (iy > 0 && iy <= IF_SIZ)){
    		din = infmap[iy-1][ix-1];
    	} else {
            ap_int<N_IFEAT> allzero;
            allzero = 0;
            din = allzero;
    	}
    	shift_reg1[ 2 * (IF_SIZ+2) + 3 - 1] = din;

    	ix++;
    	if( ix == IF_SIZ+2){
    		ix = 0;
    		iy++;
    	}


    	// enable MAC operation
    	if( k >= ((IF_SIZ+2)*2+3 - 1)){
    		w_flag++;
    		if( w_flag > (IF_SIZ+2)){
            	w_flag = 1;
            	cnt    = 0;
            }
    	}

    	// convolutional operation -----------------------------------
		ap_uint<N_OFEAT>bit_tmp = 0x1;
		ap_uint<N_OFEAT> streamOut = 0;

    	OF: for( ofeat = 0; ofeat < N_OFEAT; ofeat++){
    		int tmp = 0;
    		ap_int<20> tmp2;

            CONV_KY: for( ky = 0; ky < 3; ky++){
#pragma HLS pipeline
            	CONV_KX: for( kx = 0; kx < 3; kx++){
            		ap_int<64> bx;
            		ap_int<3> bw;

            		bx = shift_reg1[ky * (IF_SIZ+2) + kx];
            		bw = W[ofeat][ky*3+kx];

            		MAC_RGB: for( i = 0; i < 3; i++){
            			tmp2 = ap_int<20>(bx & 0xFFFFF);
            			tmp = ((bw & 0x1) == 0) ? (tmp - (int)tmp2) : (tmp + (int)tmp2);
            			bw = bw >> 1;
            			bx = bx >> 20;
            		}
            	}
            }

            // output to Streaming Buffer
            if( w_flag > 0 && w_flag <= IF_SIZ){
#pragma HLS pipeline

            	tmp += BNFb[ofeat];

            	if( tmp >= 0) streamOut = streamOut | bit_tmp;

            	bit_tmp = bit_tmp << 1;

            	cnt++;
            	if( cnt == N_OFEAT){
            		cnt = 0;

            		outfmap[oy][ox] = streamOut;

            		ox++;
            		if( ox == OF_SIZ){
            			ox = 0;
            			oy++;
            		}

            		idx++;
            	}

            }
    	}

    }
}

template< typename BIN_TYPE, typename BOUT_TYPE, int NUM_IFEAT, int NUM_OFEAT,
          int INFEAT_SIZ, int OFEAT_SIZ>
void int_conv2d_layer(
		BIN_TYPE infmap[INFEAT_SIZ][INFEAT_SIZ],
		BOUT_TYPE outfmap[OFEAT_SIZ][OFEAT_SIZ],
		ap_int<(NUMIMG)> W[NUM_OFEAT][3*3],
		ap_int<20> BNFb[NUM_OFEAT]
)
{
	int_conv2d_pipeline< BIN_TYPE, BOUT_TYPE, NUM_IFEAT, NUM_OFEAT,
		INFEAT_SIZ, OFEAT_SIZ>( infmap, outfmap, W, BNFb);
}

// -------------------------------------------------------------------------
// Maximum Pooling Layer
// -------------------------------------------------------------------------
template< typename TYPE_BIT, int FEAT_SIZ, int POOL_SIZ>
void max_pooling_layer( TYPE_BIT ftmp[FEAT_SIZ][FEAT_SIZ])
{
	int inf_x, inf_y, oy, ox;

	TYPE_BIT tmp0, tmp1, tmp2, tmp3, m;

	oy = 0;
	PY: for( inf_y = 0; inf_y < FEAT_SIZ; inf_y += 2){
		ox = 0;
		PX: for( inf_x = 0; inf_x < FEAT_SIZ; inf_x += 2){
			tmp0 = ftmp[inf_y][inf_x];
			tmp1 = ftmp[inf_y][inf_x+1];
			tmp2 = ftmp[inf_y+1][inf_x];
			tmp3 = ftmp[inf_y+1][inf_x+1];

			m = tmp0 | tmp1 | tmp2 | tmp3;
			ftmp[oy][ox] = m;
			ox++;
		}
		oy++;
	}
}

// -------------------------------------------------------------------------
// FC Layer
// -------------------------------------------------------------------------
template < int NUM_OFEAT, int NUM_INFEAT>
void fc_layer(
	ap_int<1> fc_tmp[NUM_INFEAT],
	ap_int<1> lW[NUM_OFEAT][NUM_INFEAT],
	ap_int<16> b_BNFb[NUM_OFEAT],
	int fc_result[(MAX_DENSE_SIZ)]
)
{
	int ofeat, tmp, infeat;

	FC_O: for( ofeat = 0; ofeat < NUM_OFEAT; ofeat++){
#pragma HLS LOOP_FLATTEN off
		tmp = 0;

		FC_I: for( infeat = 0; infeat < NUM_INFEAT; infeat++){
#pragma HLS pipeline
			ap_int<1> bw, bx, xnor;

			bw = lW[ofeat][infeat];
			bx = fc_tmp[infeat];
			xnor = ~(bw ^ bx);

			tmp += (xnor == 0) ? -1 : +1;
		}

		fc_result[ofeat] = tmp + b_BNFb[ofeat];
	}
}

// -------------------------------------------------------------------------
// Binarized CNN Kernel
// -------------------------------------------------------------------------
#ifdef __SDSCC__
#pragma SDS data access_pattern(t_in_img: SEQUENTIAL)
#pragma SDS data zero_copy(t_in_img[0:(IMGSIZ)*(IMGSIZ)])
#endif
void kernel(
#ifdef __SDSCC__
        ap_int<64> t_in_img[(IMGSIZ)*(IMGSIZ)],
        int fc_result[10]
#else 
        ap_int<64> t_in_img[(IMGSIZ)*(IMGSIZ)],
        int fc_result[10]
#endif
)
{
	ap_int<(MAX_BCONV_WIDTH)> fb_tmp[(IMGSIZ)][(IMGSIZ)];
	ap_int<1> fc_tmp[(MAX_DENSE_SIZ)];
	ap_int<64> in_img[(IMGSIZ)][(IMGSIZ)];

	int y, x, of, layer, bin_layer_idx;
(DEF_CNN_PARAMETER)

	for( y = 0; y < (IMGSIZ); y++){
		for( x = 0; x < (IMGSIZ); x++){
			in_img[y][x] = t_in_img[y*(IMGSIZ)+x];
		}
	}

#pragma HLS INLINE

    bin_layer_idx = 1;
	BCONV: for( layer = 0; layer < (NUM_LAYER); layer++){
		switch(layer){
(DEF_CNN_LAYER)
		}
	}
}

//--------------------------------------------------------------------
// Top Function for a Binarized CNN
//--------------------------------------------------------------------
#ifdef __SDSCC__
#pragma SDS data access_pattern(t_bin_convW: SEQUENTIAL)
#pragma SDS data access_pattern(t_BNFb: SEQUENTIAL)
#pragma SDS data access_pattern(t_in_img: SEQUENTIAL)
#pragma SDS data zero_copy(t_bin_convW[0:(WEIGHT_SIZ)])
#pragma SDS data zero_copy(t_BNFb[0:(BIAS_SIZ)])
#pragma SDS data zero_copy(t_in_img[0:(IMGSIZ)*(IMGSIZ)])
#endif
void BinCNN(
#ifdef __SDSCC__
        int *t_bin_convW,
        int *t_BNFb,
        ap_int<64> t_in_img[(IMGSIZ)*(IMGSIZ)],
        int fc_result[(OUT_DENSE_SIZ)],
        int init
#else 
        int t_bin_convW[(WEIGHT_SIZ)],
        int t_BNFb[(BIAS_SIZ)],
        ap_int<64> t_in_img[(IMGSIZ)*(IMGSIZ)],
        int fc_result[(OUT_DENSE_SIZ)],
        int init
#endif
)
{
/*
#pragma HLS INTERFACE s_axilite register port=t_bin_convW bundle=slv0
#pragma HLS INTERFACE s_axilite register port=t_BNFb bundle=slv0
#pragma HLS INTERFACE s_axilite register port=t_in_img bundle=slv0
#pragma HLS INTERFACE s_axilite register port=fc_result bundle=slv0
#pragma HLS INTERFACE s_axilite register port=init bundle=slv0
#pragma HLS INTERFACE s_axilite register port=return bundle=slv0
*/
	if( init == 1)
		setup( t_bin_convW, t_BNFb);
	else
		kernel( t_in_img, fc_result);
}

// ------------------------------------------------------------------
// END OF PROGRAM
// ------------------------------------------------------------------
