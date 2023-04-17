
.. Places parent toc into the sidebar

:parenttoc: True

=========================
Getting started
=========================

.. currentmodule:: emlearn

Prerequisites
===========================

You need to have installed **Python** (version 3.6+),
and a **C99 compiler** for your platform (GCC/Clang/MSVC).

On Windows, the Windows Subsystem for Linux (WSL) is recommended,
but MSCV and cmd.exe/Powershell can also be used.

Install scikit-learn 
===========================

In this example, **scikit-learn** is used to train the models.

.. code-block:: console

    pip install scikit-learn

Install emlearn
===========================

**emlearn** will be used to convert the scikit-learn models to C code.

.. code-block:: console

    pip install emlearn


Create model in Python
===========================

We will train a simple model to learn the XOR function.
The same steps will be used for model of any complexity.
Copy and save this as file ``xor_train.py``.

.. literalinclude:: helloworld_xor/xor_train.py
   :language: python
   :emphasize-lines: 1,21-24
   :linenos:

Run the script

.. code-block:: console

    python xor_train.py

It will generate a file ``xor_model.h`` containing the C code for our model.

Use in C code 
========================

To run our model we use a simple C program that
takes data on the commandline, and prints out the detected class.

Copy and save this as file ``xor_host.c``.

.. literalinclude:: helloworld_xor/xor_host.c
   :language: c
   :emphasize-lines: 1,18-19
   :linenos:


On Linux / MacOS / WSL with GCC

.. code-block:: console

    export EMLEARN_INCLUDE_DIR=`python -c 'import emlearn; print(emlearn.includedir)'`
    gcc -o xor_host xor_host.c -I${EMLEARN_INCLUDE_DIR}

On Windows

.. code-block:: console

    cl xor_host.c /link /out:xor_host.exe

Try it out 
========================

In our training data input values above ``0.5`` is considered "true".
So for the XOR function, if **one and only one** of the values is above ``0.5``, should get class **1** as output - else class **0**. 

The following should output 1

.. code-block:: console

    ./xor_host 0.6 0.0
    ./xor_host 0.1 0.7

The following should output 0

.. code-block:: console

    ./xor_host 0.8 0.7
    ./xor_host 0.8 0.7
    ./xor_host 0.0 0.0

Next
========

Now you have the emlearn tool running on your system,
and it can be used to convert the models that you are interested in.

.. TODO: link to a hardware getting-started. Arduino et.c.


