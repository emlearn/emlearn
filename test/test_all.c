
/* Entry-point for running C tests */

#define EML_DEBUG

#include "test_array.c"
#include <unity.c>

void
setUp(void)
{
    // set stuff up here
}

void
tearDown(void)
{
    // clean stuff up here
}

// gcc -std=c99 -o run_tests test/test_all.c -g -I./emlearn -I./test/Unity/src/ -lm && ./run_tests
int main(void)
{
    UNITY_BEGIN();

    // Run the different test modules
    test_eml_array();

    return UNITY_END();
}
