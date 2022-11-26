
# TODO
- Integrate into docs / examples.
Might be that some of this should go into a user-guide section/chapter
on "Optimizing models".
Allows the example to have a more sparse/direct texts
And also allows to split into several examples, with more singular focus (but referring to user guide for the whole)


## Goals for efficient models

When creating models for efficient inference on constrained hardware target,
there are generally 3 concerns:

- Time to make a prediction (CPU).
- Working memory required (RAM).
- Storage required for model (FLASH/program memory).

These tend to heavily correlated,
though some optimization strategies may influence one aspect more than the others.
And in a few cases, it may be possible to trade one against the other.

Sometimes it is also desirable to optimize the prediction latency.
This is usually highly dependent on the feature extraction,
and so will not be covered here. 

## Optimization of features

Tree-based models are pretty good at ignorning less-useful features.
Therefore it is not generally neccesary to do separate feature selection step.
But one might want to remove features that are completely useless or redundant, if present.

Creating new features using feature engineering on the other hand can have a very large impact,
and should be considered in addition to optimizing the classifier.
This tends to be very problem/task dependent and is outside of the scope of this example.
But for inspiration, see for example
[Energy-efficient activity recognition framework using wearable accelerometers](https://www.mdpi.com/2079-9292/10/21/2640/htm)
where tree-based models outperform Convolutional Neural Networks.


## Hyperparameter optimization

Limit Width and depth.
There are several hyperparameters which allow limiting depth.

TODO: reference example


## Tradeoffs in predictive performance and model costs

A larger model will generally have higher predictive performance, but need more CPU/RAM/storage.
This leads to a tradeoff, and different applications may chose different operating points. 
We may try to find a set of [Pareto optimal](https://en.wikipedia.org/wiki/Pareto_efficiency) model alternatives.

TODO: integrate this with notebook

## Quantization

TODO: show in notebook how to do



## Measuring model costs

The best measurements will be by running the model on the target hardware.
However, having the hardware in the loop makes it slow to evaluate a large set of models.
Therefore this is often done at the last stage, only for verification.

The simplest alternative is to use the properties of the model.

Slightly better is to run the generated model on the target hardware.
One can also compile for the target platform, and inspect the code size and RAM size in that way.

### TODO
TODO: setup tooling for measuring FLASH/SRAM using GCC
TODO: setup tooling for running benchmark
TODO: include this in example notebook

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
Like for a bare-metal Raspeberry PI arm-linux-gnueabihf-gcc -mcpu=cortex-a7 -marm

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
Lots of details to this, so would want to use something else to make this managable.
Candidates: Zephyr, PlatformIO, Arduino??



