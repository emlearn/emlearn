
/* Entry-point for running C tests */

#define EML_LOG_ENABLE 1
#define EML_DEBUG

#include <eml_log.h>

#include "test_array.c"
#include "test_neighbors.c"
#include "test_quantizer.c"
#include "test_trees.c"
#include "test_net.c"

#include <unity.c>

// Declare the different modules
typedef void (*TestModuleFunction)(void);

typedef struct _TestModule {
    const char * name;
    TestModuleFunction func;
} TestModule;

#define TEST_MODULES 5
TestModule test_modules[TEST_MODULES] = {
    { "array", test_eml_array },
    { "neighbors", test_eml_neighbors },
    { "quantizer", test_eml_quantizer },
    { "net", test_eml_net },
    { "trees", test_eml_trees },
};

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

int
run_tests(const char *tests)
{
    UNITY_BEGIN();

    // Run the different test modules
    const char *tests_enabled = tests;
    const char *env = getenv("EMLEARN_TEST_MODULES");
    if (!tests_enabled) {
        tests_enabled = env;
    }

    EML_LOG_BEGIN("run-tests");
    EML_LOG_ADD("env", env);
    EML_LOG_ADD("enabled", tests_enabled);
    EML_LOG_END();

    for (int i=0; i<TEST_MODULES; i++) {
        const TestModule *mod = &test_modules[i];

        bool run_test = true;
        if (tests_enabled != NULL) {
            run_test = strstr(mod->name, tests_enabled) != NULL;
        }

        if (run_test) {
            mod->func();
        }
    }

    return UNITY_END();
}

// gcc -std=c99 -o run_tests test/test_all.c -g -I./emlearn -I./test/Unity/src/ -lm && ./run_tests
int main(void)
{
    run_tests(NULL);
}
