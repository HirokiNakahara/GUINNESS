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
);

//--------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------
int main( int argc, char *argv[])
{
    ap_int<64> *t_tmp_img;
    t_tmp_img = (ap_int<64> *)sds_alloc(((IMGSIZ)*(IMGSIZ))*sizeof(ap_int<64>));

    int fc_result[(OUT_DENSE_SIZ)];
    int rgb, y, x, i, offset;

    // copy input image to f1
    for( y = 0; y < (IMGSIZ); y++){
    	for( x = 0; x < (IMGSIZ); x++){
    		t_tmp_img[y*(IMGSIZ)+x] = 0;
        }
    }

    // ------------------------------------------------------------------
    printf("load weights\n");
    int *t_bin_convW;
	int *t_BNFb;
	t_bin_convW = (int *)sds_alloc(((WEIGHT_SIZ))*sizeof(int));
	t_BNFb   = (int *)sds_alloc(((BIAS_SIZ))*sizeof(int));

	int of, inf, d_value;
	FILE *fp;
	char line[256];

(READ_BIAS_MEM)

(READ_WEIGHT_MEM)

    printf("setup... \n");
	BinCNN( t_bin_convW, t_BNFb, t_tmp_img, fc_result, 1);

    char image_name[256];
    int cnt;

#ifdef __SDSCC__
    sscanf( argv[1], "%s", image_name); // 1st argument: test image (text file)
    sscanf( argv[2], "%d", &cnt); // 2nd argument: # of inferences 
#else 
    sprintf( image_name, "test_img.txt");
    cnt = 1;
#endif


    int pixel;
    printf("LOAD TESTBENCH %s ... ", image_name);
    if( (fp = fopen(image_name, "r")) == NULL)fprintf(stderr,"CANNOT OPEN\n");
    for( y = 0; y < (IMGSIZ); y++){
        for( x = 0; x < (IMGSIZ); x++){
            ap_int<64>tmp = 0;
            for( rgb = (NUMIMG) - 1; rgb >= 0 ; rgb--){
                if( fgets( line, 256, fp) == NULL)
                    fprintf(stderr,"EMPTY FILE READ\n"); 
                sscanf( line, "%d", &d_value);

                tmp = tmp << 20;

                pixel = d_value;
                tmp |= ( pixel & 0xFFFFF);
            }
            t_tmp_img[ y * (IMGSIZ) + x] = tmp;
        }
    }
    printf("OK\n");
    fclose(fp);

    printf("Inference %d times ... ", cnt);
    for( i = 0; i < cnt; i++){
        BinCNN( t_bin_convW, t_BNFb, t_tmp_img, fc_result, 0);
    }
    printf("OK\n");

    printf("Result\n");
    for( i = 0; i < (OUT_DENSE_SIZ); i++)printf("%5d ", fc_result[i]);
    printf("\n");

    sds_free( t_tmp_img); sds_free( t_bin_convW); sds_free( t_BNFb);

    return 0;
}

// ------------------------------------------------------------------
// END OF PROGRAM
// ------------------------------------------------------------------
