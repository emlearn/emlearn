
#include <stdlib.h> // stdod
#include <stdio.h>
#include <math.h>

void
polar_transform(const float xy[2], float polar[2])
{
    const float eps = 1e-12;
    const float x = xy[0]+eps;
    const float y = xy[1]+eps;

    float angle = atanf(y/x);
    if (x < 0.0) {
        // 2nd or 3rd quadrant
        angle += M_PI;
    }
    float magnitude = sqrtf(x*x + y*y); 

    polar[0] = angle;
    polar[1] = magnitude;
}

int
main(int argc, const char *argv[])
{

    if (argc != 3) {
        fprintf(stderr, "ERROR: wrong number of arguments %d\n", argc);
        return -1;
    }
    printf("input %s %s %s \n", argv[0], argv[1], argv[2]);

    const float x = strtod(argv[1], NULL);
    const float y = strtod(argv[2], NULL);

    float out[2] = { 0.0, 0.0 };
    const float xy[] = { x, y };
    polar_transform(xy, out);
    printf("polar-transform x=%.2f y=%.2f angle=%.2f magnitude=%.2f \n",
        xy[0], xy[1], out[0], out[1]
    );

}

// take paths to .CSV files - one for input, one for output
// Python Transformer. Takes path to .c program, which transforms a CSV input to CSV output
// provide tools for implementing this
// Bring from eml_test.h to eml_csv.h


NOTE: might also need to support linking to external libs
from emlearn.transform import ProgramTransformer
ProgramTransformer(path='my_code.c')

EmlError err = eml_transform_file(argv[1], argv[1], function)
return -err;


