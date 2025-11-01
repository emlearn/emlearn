.. zephyr:code-sample:: sensor_reader
   :name: Sensor readout using 
   :relevant-api: sensor_interface

   Get accelerometer and gyroscope data from an LSM6DSL sensor.

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

Building and Running
********************


Building on ArgonKey board
==========================

.. zephyr-app-commands::
   :zephyr-app: samples/emlearn/sensor_reader
   :host-os: unix
   :board: xiao_ble/nrf52840/sense
   :goals: build
   :compact:

Sample Output
=============

.. code-block:: console

    LSM6DSL sensor samples:

    XXX 


