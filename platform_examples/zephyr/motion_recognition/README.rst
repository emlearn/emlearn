Motion classification
********

Status
************************

**Work in progress**.

Implemented

* Reading sensor data from LSM6DSL
* Storing sensor data to USB disk, as CSV files
* Preprocessing code for motion, with IIR gravity estimation and FFT analysis of motion 
* Generating the model/preprocessing code as part of west build, using CMake targets
* Tool for running preprocessing/model on CSV files, for training and validation on PC

TODO complete example

* Train. Integrate preprocessing tool into training pipeline.
har_train.py from https://github.com/emlearn/emlearn-micropython/tree/master/examples/har_trees 
* Live. Also run the model on extracted features, print results

Improvements

* Validation. Selfcheck runs on file with sensor data,
outputs model results, and verifies the results
* Reading. Support reading LSM6DSL using FIFO
* Reading. Support overlapped windows
* Storing. Support the external SPI flash on XIAO BLE sense (2MB) 
* Storing. Support NPY format instead of CSV, much more space efficient


Requirements
************

Sample has been tested on XIAO BLE sense NRF52840,
using the on-board LSM6DSL sensor.
Sample has been tested using Linux as the build system.
Should work also with Mac OS or Windows Subsystem for Linux (WSL).


References
**********

- LSM6DSL https://www.st.com/en/mems-and-sensors/lsm6dsl.html
- emlearn https://github.com/emlearn/emlearn


Building and Running
********************

Need to have Python3 and install some dependencies

.. code-block:: console

    python3 -m venv venv
    source venv/bin/activate
    pip install emlearn


Building on XIAO BLE Sense NRF52840 board
==========================

.. code-block:: console

    west build --board xiao_ble/nrf52840/sense ./motion_recognition/ -- -DEXTRA_DTC_OVERLAY_FILE="flashdisk.overlay" -DCONFIG_APP_MSC_STORAGE_FLASH_FATFS=y -DCONFIG_DISK_DRIVER_SDMMC=n


Sample Output
=============

.. code-block:: console

    TODO

