.. zephyr:code-sample:: sensor_reader
   :name: Motion recognition using emlearn
   :relevant-api: sensor_interface

   Classify motion using machine learning

Overview
********

TODO

Requirements
************

This sample uses the LSM6DSL sensor controlled using the I2C or SPI interface.
It has been tested on XIAO BLE sense.

References
**********

- LSM6DSL https://www.st.com/en/mems-and-sensors/lsm6dsl.html
- emlearn https://github.com/emlearn/emlearn

Building and Running
********************


Building on XIAO BLE Sense NRF52840 board
==========================

.. code-block:: console

    west build --board xiao_ble/nrf52840/sense ./motion_recognition/



Sample Output
=============

.. code-block:: console

    LSM6DSL sensor samples:

    XXX


