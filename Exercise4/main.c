#include "tiffio.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    TIFF* tif = TIFFOpen(argv[1], "r");
    if (tif) {
        uint32 width, height;
        uint16 samplesPerPixel, bitsPerSample;
        char msg[1024];
        
        // 读取图像的宽度和高度
        if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) != 1) {
            printf("Failed to get image width\n");
            return EXIT_FAILURE;
        }
        if (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) != 1) {
            printf("Failed to get image height\n");
            return EXIT_FAILURE;
        }
        
        // 读取图像的位深度和样本数量
        if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel) != 1) {
            printf("Failed to get samples per pixel\n");
            return EXIT_FAILURE;
        }
        if (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample) != 1) {
            printf("Failed to get bits per sample\n");
            return EXIT_FAILURE;
        }
        
        // 打印图像信息
        sprintf(msg, "Image info:\n\tWidth: %d\n\tHeight: %d\n\tSamples per pixel: %d\n\tBits per sample: %d\n", width, height, samplesPerPixel, bitsPerSample);
        printf("%s", msg);
        
        // 关闭文件
        TIFFClose(tif);
    } else {
        printf("Cannot open file %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

