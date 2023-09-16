
.. Places parent toc into the sidebar

:parenttoc: True

.. _getting_started_host:

=========================
Getting started on PC (Linux/MacOS/Windows)
=========================

.. currentmodule:: emlearn

emlearn models work anywhere there is a C99 compiler available.
This includes common desktop platforms such as Linux, Mac OS, Windows, etc.
Since you need such a host platform to develop the Python machine-learning,
it is convenient also to do the first tests of the model on the host.

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

On Windows with cmd.exe

.. code-block:: console

    python -c "import emlearn; print(emlearn.includedir)"
    
    set EMLEARN_INCLUDE_DIR=    output from above command
    
    cl xor_host.c /I %EMLEARN_INCLUDE_DIR% /link /out:xor_host.exe

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
    ./xor_host 0.0 0.0

Next
========

Now you have the emlearn tool running on your system,
and it can be used to convert the models that you are interested in.

You may be interested in trying it out on a hardware device.
See for example :doc:`getting_started_arduino`.

