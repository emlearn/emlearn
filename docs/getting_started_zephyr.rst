
.. Places parent toc into the sidebar
:parenttoc: True

.. _getting_started_zephyr:

=========================
Getting started on Zephyr RTOS
=========================

.. currentmodule:: emlearn

emlearn works on any platform that has C99 support.
`Zephyr <https://www.zephyrproject.org/>`_ is a modern, professional, open-source RTOS.
This guide shows you how to use emlearn on that platform.

Prerequisites
===========================

Ensure that you have **emlearn** setup as per the :doc:`getting_started_host`.

You will need to have Zephyr installed already.
Follow the `Zephyr Getting Started Guide <https://docs.zephyrproject.org/latest/develop/getting_started/index.html>`_.
In particular you will a functional setup that includes the Zephyr SDK, and the ``west`` build tool.

You will need to have a Zephyr project setup.
If you do not already have one, either use one of the samples in Zephyr as a base,
or the `Zephyr example application <https://github.com/zephyrproject-rtos/example-application>`_.

No particular hardware is needed. Any board supported by Zepyr should work.
In the example below, a board designed for QEMU simulator is used.
Support for this is included in the standard Zephyr SDK setup.
But if you have gone for a minimal setup, it may need to be installed manually.
The QEMU board specifically requires the ``arm-zephyr-eabi`` compiler toolchain,
and the ``cmsis`` and ``hal_nordic`` Zephyr modules.

Add emlearn to your Zephyr project
========================

emlearn provides the neccesary metadata to be used a Zephyr module (since Feburary 2024).
To include the module, add an entry to the ``projects`` section of your ``west.yml`` configuration.

.. code-block:: yaml

    projects:
      #... exiting projects
      - name: emlearn
        url: https://github.com/emlearn/emlearn
        revision: master

To make sure it worked correctly, you can run

.. code-block:: console

    west update emlearn

This should now pull the emlearn git repository into your workspace.


Enable emlearn in your Zephyr project
========================

Once you have the emlearn Zephyr module, you must enable it in your application.
You can add this manually to your configuration file.
This file is typically called ``prj.conf``.

.. code-block::

    CONFIG_EMLEARN=y


Create model in Python
===========================

We will train a simple model to learn the XOR function.
Copy and save this as file ``xor_train.py``.

.. literalinclude:: helloworld_xor/xor_train.py
   :language: python
   :emphasize-lines: 1,21-24
   :linenos:

Run the script

.. code-block:: console

    python xor_train.py

It will generate a file ``xor_model.h`` containing the C code for our model.
Copy this file into your ``src/`` directory.


Example application code
========================

Here is some example code for calling the XOR model on some inputs.

.. literalinclude:: ../platform_examples/zephyr/helloworld_xor/src/main.c
   :language: c
   :emphasize-lines: 7,20,28
   :linenos:


Build and run in simulator
========================

We are assuming that your code is called `helloworld_xor`.

.. code-block:: console

    west build --board qemu_cortex_m3 ./helloworld_xor/ -t run

This should result in output similar to this:

.. code-block:: console

    ...
    -- west build: running target run
    ...
    To exit from QEMU enter: 'CTRL+a, x'[QEMU] CPU: cortex-m3
    ...
    *** Booting Zephyr OS build zephyr-v3.5.0 ***
    xor(0,0) = 0
    xor(1,0) = 1
    xor(0,1) = 1
    xor(1,1) = 0



