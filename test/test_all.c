
/* Entry-point for running C tests */

#define EML_LOG_ENABLE 1
#define EML_DEBUG

#include <eml_log.h>

#include "test_array.c"
#include "test_signal_windower.c"

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
    test_eml_signal_windower();

    return UNITY_END();
}
