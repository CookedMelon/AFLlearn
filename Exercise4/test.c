#include "tiffio.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    TIFF* tif = TIFFOpen(argv[1], "r");
    if (tif) {
	return 0;
        return EXIT_FAILURE;
    } else {
        printf("Cannot open file %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    return 0;
}
