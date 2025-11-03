#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include "image.h"
#include <time.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[] = {
    {{0,-1,0},{-1,4,-1},{0,-1,0}}, // EDGE
    {{0,-1,0},{-1,5,-1},{0,-1,0}}, // SHARPEN
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}}, // BLUR
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}}, // GAUSE_BLUR
    {{-2,-1,0},{-1,1,1},{0,1,2}}, // EMBOSS
    {{0,0,0},{0,1,0},{0,0,0}} // IDENTITY
};

//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//          y: The y coordinate of the pixel
//          bit: The color channel being manipulated
//          algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my;
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    double val =
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)] +
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)] +
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (uint8_t)val;
}

// Worker that processes a range of rows
typedef struct {
    Image *src;
    Image *dst;
    Matrix kernel;
    int row_start; // inclusive
    int row_end;   // exclusive
} ThreadData;

void *thread_convolute(void *arg){
    ThreadData *td = (ThreadData*)arg;
    Image *src = td->src;
    Image *dst = td->dst;
    for (int row = td->row_start; row < td->row_end; ++row){
        for (int pix = 0; pix < src->width; ++pix){
            for (int bit = 0; bit < src->bpp; ++bit){
                dst->data[Index(pix,row,src->width,bit,src->bpp)] =
                    getPixelValue(src,pix,row,bit,td->kernel);
            }
        }
    }
    return NULL;
}

//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image_pthreads <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc,char** argv){
    if (argc!=3) return Usage();
    char* fileName=argv[1];
    enum KernelTypes type=GetKernelType(argv[2]);

    stbi_set_flip_vertically_on_load(0);
    Image srcImage,dstImage;
    srcImage.data = stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }
    dstImage.bpp = srcImage.bpp;
    dstImage.width = srcImage.width;
    dstImage.height = srcImage.height;
    dstImage.data = malloc((size_t)dstImage.width * dstImage.height * dstImage.bpp);
    if (!dstImage.data){
        fprintf(stderr,"malloc failed\n");
        stbi_image_free(srcImage.data);
        return -1;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Number of threads
    int num_threads = 4; 
    if (num_threads < 1) num_threads = 1;
    pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
    ThreadData *td = malloc(sizeof(ThreadData) * num_threads);

    int rows_per = srcImage.height / num_threads;
    int extra = srcImage.height % num_threads;
    int current_row = 0;
    for (int i = 0; i < num_threads; ++i){
        int start = current_row;
        int end = start + rows_per + (i < extra ? 1 : 0);
        current_row = end;
        td[i].src = &srcImage;
        td[i].dst = &dstImage;
        memcpy(td[i].kernel, algorithms[type], sizeof(Matrix));
        td[i].row_start = start;
        td[i].row_end = end;
        if (pthread_create(&threads[i], NULL, thread_convolute, &td[i]) != 0){
            fprintf(stderr, "Error creating thread %d\n", i);
            thread_convolute(&td[i]);
        }
    }

    for (int i = 0; i < num_threads; ++i){
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1e9;
    printf("Processing time: %.4f seconds\n", elapsed);

    stbi_write_png("output.png", dstImage.width, dstImage.height, dstImage.bpp, dstImage.data, dstImage.bpp*dstImage.width);

    free(threads);
    free(td);
    stbi_image_free(srcImage.data);
    free(dstImage.data);
    return 0;
}
