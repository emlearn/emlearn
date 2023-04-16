
.. Places parent toc into the sidebar

:parenttoc: True

=========================
Model optimization
=========================

.. currentmodule:: emlearn

Goals for efficient models
===========================

When designing for efficient inference on a constrained hardware target,
there are generally 3 concerns:

- Time to make a prediction (CPU).
- Working memory required (RAM).
- Storage required for model (FLASH/program memory).

These tend to heavily correlated,
though some optimization strategies may influence one aspect more than the others.
And in a few cases, it may be possible to trade one against the other.

Sometimes it is also desirable to optimize the detection latency:
the time it takes the system to detect something happening.
This is mostly often highly dependent on the feature extraction, and so will not be covered here.

Measuring model costs
========================

The best measurements will be by running the model on the target hardware.
However, having the hardware in the loop makes it slow to evaluate a large set of models.
Therefore this is often done at the last stage, only for verification.

The simplest alternative is to use the properties of the model.

Slightly better is to run the generated model on the target hardware.
One can also compile for the target platform, and inspect the code size and RAM size in that way.
