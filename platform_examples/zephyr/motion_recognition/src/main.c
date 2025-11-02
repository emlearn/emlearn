/*
 * Copyright (c) 2018 STMicroelectronics
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <stdio.h>
#include <zephyr/sys/util.h>

// 
#include "sample_usbd.h"

int usb_disk_setup(void);

// Application includes
#include "sensor_reader.h"
#include "motion_preprocessing.h"
#include "motion_gravity_lowpass.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

// Configuration
#define SAMPLERATE 52
#define WINDOW_LENGTH 50
#define HOP_LENGTH 25

#define N_CHANNELS 6
enum sensor_channel sensor_reader_channels[N_CHANNELS] = {
	SENSOR_CHAN_ACCEL_X,
	SENSOR_CHAN_ACCEL_Y,
	SENSOR_CHAN_ACCEL_Z,
	SENSOR_CHAN_GYRO_X,
	SENSOR_CHAN_GYRO_Y,
	SENSOR_CHAN_GYRO_Z,
};


#define MOTION_MODEL_CLASSES 0
#define FEATURE_COLUMNS_LENGTH (1+motion_features_length+1+MOTION_MODEL_CLASSES+MOTION_FFT_LENGTH)
float feature_values[FEATURE_COLUMNS_LENGTH];


// Reader internals
float sensor_reader_output_buffer[WINDOW_LENGTH*N_CHANNELS];
float sensor_reader_new_buffer[HOP_LENGTH*N_CHANNELS];

#define SENSOR_READER_STACK_SIZE 1000
K_THREAD_STACK_DEFINE(sensor_reader_stack, SENSOR_READER_STACK_SIZE);
struct k_thread sensor_reader_thread;

K_MSGQ_DEFINE(sensor_reader_queue, sizeof(struct sensor_chunk_msg), 1, 1);


int
setup_sensor(const struct device *const lsm6dsl_dev)
{
	struct sensor_value odr_attr;
	if (!device_is_ready(lsm6dsl_dev)) {
		printk("sensor: device not ready.\n");
		return -1;
	}

	/* set accel/gyro sampling frequency */
	odr_attr.val1 = SAMPLERATE;
	odr_attr.val2 = 0;

	if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_ACCEL_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
		printk("Cannot set sampling frequency for accelerometer.\n");
		return -2;
	}

	if (sensor_attr_set(lsm6dsl_dev, SENSOR_CHAN_GYRO_XYZ,
			    SENSOR_ATTR_SAMPLING_FREQUENCY, &odr_attr) < 0) {
		printk("Cannot set sampling frequency for gyro.\n");
		return -3;
	}

    return 0;
}


// Large, use global and not stack
struct motion_preprocessor _preprocessor;

int main(void) {

	printk("Start USB setup.\n");
    // Setup USB disk
    const int usb_err = usb_disk_setup();
    if (usb_err) {
		LOG_ERR("Failed to setup USB disk");
    } else {
		LOG_ERR("Now in USB mass storage mode");
    }

    struct sensor_chunk_msg chunk;

	const struct device *const lsm6dsl_dev = DEVICE_DT_GET_ONE(st_lsm6dsl);

    struct sensor_chunk_reader reader = {
        .samplerate = SAMPLERATE,
        .window_length = WINDOW_LENGTH,
        .hop_length = HOP_LENGTH,
        .thread = &sensor_reader_thread,
        .thread_priority = 10,
        .read_samples = sensor_reader_new_buffer,
        .read_samples_index = 0,
        .output_buffer = sensor_reader_output_buffer,
        .output_buffer_index = 0,
        .n_channels = N_CHANNELS,
        .channels = sensor_reader_channels,
        .dev = lsm6dsl_dev,
        .queue = &sensor_reader_queue,
        .stack = sensor_reader_stack,
        .stack_size = SENSOR_READER_STACK_SIZE,
        .fetch_errors = 0,
        .get_errors = 0,
        .put_errors = 0
    };

    struct motion_preprocessor *preprocessor = &_preprocessor;

    const int init_err = motion_preprocessor_init(preprocessor, SAMPLERATE, WINDOW_LENGTH);
    if (init_err != 0) {
        fprintf(stderr, "preprocess init error %d\n", init_err);
        return -2;
    }

#if 1
    const int gravity_err = motion_preprocessor_set_gravity_lowpass(preprocessor,
        gravity_lowpass_values, gravity_lowpass_length);
    if (gravity_err != 0) {
        fprintf(stderr, "lowpass config error %d\n", gravity_err);
        return 2;
    }
#endif

    const int fft_config_err = \
        motion_preprocessor_set_fft_features(preprocessor, 1, 10);
    if (fft_config_err != 0) {
        fprintf(stderr, "FFT config error %d\n", fft_config_err);
        return 2;
    }

    if (!device_is_ready(lsm6dsl_dev)) {
        printk("sensor: device %s not ready.\n", lsm6dsl_dev->name);
        return 0;
    }

    // Setup sensor
    setup_sensor(lsm6dsl_dev);

    const int n_features = motion_preprocessor_get_feature_length(preprocessor);
    if (n_features < 0) {
        return 2;
    }

    // Start high-priority thread for collecting data
    sensor_chunk_reader_start(&reader);

    int iteration = 0;
    float previous_input = 0.0f;
    while (1) {
        const float uptime = k_uptime_get() / 1000.0;

        // check for new data
        const int get_error = k_msgq_get(reader.queue, &chunk, K_NO_WAIT);
        if (get_error == 0) {

            const float dt = uptime - previous_input;

            //printk("process-chunk length=%d \n", chunk.length);
            const int run_status = \
                motion_preprocessor_run(preprocessor, chunk.buffer, chunk.length);

            // Log extracted features
            const int copy_err = \
                motion_preprocessor_get_features(preprocessor, feature_values, FEATURE_COLUMNS_LENGTH);

            printk("features run_err=%d copy_err=%d l=%d dt=%.3f time=%.3f | ",
                run_status, copy_err, chunk.length, (double)dt, (double)uptime);
            for (int i=0; i<n_features; i++) {
                printk("%.4f ", (double)feature_values[i]);
            }
            printk("\n");

            // TODO: run through ML model, print outputs
            previous_input = uptime;
        }

        iteration += 1;
	    k_msleep(100);
    }

    return 0;
}

