
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

You will also need to have the Zephyr toolchain <https://www.arduino.cc/en/software>`_ installed.

Any board supported by Zepyr should work.
The code can also be ran in a QEMU simulator, using Zephyr board support for QEMU targets.

Add emlearn to your Zephyr project
========================

emlearn provides the neccesary metadata to be used a Zephyr module.

To include the module, add an entry to the ``projects`` in your ``west.yml`` configuration.

  projects:
    #... exiting projects
    - name: emlearn
      url: https://github.com/emlearn/emlearn
      revision: master

To make sure it worked correctly, you can run ``west update``.

.. code-block:: console

    west update emlearn

This should now pull the emlearn git repostitory.

TODO: illustrate example output

Enable emlearn in your Zephyr project
========================

Once you have the emlearn Zephyr module, you must enable it in your application.

prj.conf
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


Example application code
========================

Replace the sketch code in the Arduino IDE with the contents of the following.

.. literalinclude:: helloworld_xor/helloworld_xor.ino
   :language: c
   :emphasize-lines: 1,18-19
   :linenos:

Wait with compiling, as we will need the generated C code from the next step.



Build and upload the code
========================

.. code-block:: console

    west build


Test it out 
========================





