
# Optimizing tree models

## Documentation

In docs/tree_based_models.rst

## Improvements

### TODO

- TODO: setup tooling for measuring FLASH/SRAM using GCC
- TODO: setup tooling for running benchmark
- TODO: include this in example notebook

#### Size measurement tooling.

Generate code for
A) complete binary without the model
B) complete binary with the model

Ideally compile for the target platform.
Include at least arm-none-eabi.
Secondary targets, xtensa/ESP, avr8/Arduino

Will need linker definitions...
Need to find a good, simple and clear base.
Might be useful to have one with huge memory areas (bigger than what exists on typical MCU).
Like for a bare-metal Raspberry PI arm-linux-gnueabihf-gcc -mcpu=cortex-a7 -marm

Use GCC to extract the sizes. Used progmem and flash
Compute model resource consumption as difference from no-model reference.

#### Benchmark tooling

Generate code that includes
A) the model(s) under test
B) an evaluation dataset
C) benchmark code, that runs the model on the dataset several times and measures execution time 
Compile this for the host.
Run this on the host, extracting the results
Since compile and run has some overhead - and models are expected to be small, support evaluating multiple at a time.

Benchmark stopping threshold can be either n_iterations, or fixed time period, or until variance stabilizes?

Output should be one entry per test execution run?
For example CSV
run,model,time_ms
0,modelA,300
...

Later: easy way to run with simulator.
Using QEMU via Zephyr tooling maybe?
Alternative to QEMU is to use GDB "target sim"
Probably neither are very realistic.
But at least it offers a way to easily test the entire pipeline and tooling on proper architecture.

Later: easy way to run on real target
Ideally fully automated. Meaning that device lookup, flashing, starting, output is all automatic.
Whether this is feasible probably depends on hardware.
Lots of details to this, so would want to use something else to make this manageable.
Candidates: Zephyr, PlatformIO, Arduino??



