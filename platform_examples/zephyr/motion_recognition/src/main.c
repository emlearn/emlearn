/*
 * Copyright (c) 2018 STMicroelectronics
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <stdio.h>
#include <zephyr/sys/util.h>

#include "sensor_reader.h"
#include "preprocessing.h"

// Configuration
#define SAMPLERATE 104
#define WINDOW_LENGTH 100
#define HOP_LENGTH 50

#define N_CHANNELS 6
enum sensor_channel sensor_reader_channels[N_CHANNELS] = {
	SENSOR_CHAN_ACCEL_X,
	SENSOR_CHAN_ACCEL_Y,
	SENSOR_CHAN_ACCEL_Z,
	SENSOR_CHAN_GYRO_X,
	SENSOR_CHAN_GYRO_Y,
	SENSOR_CHAN_GYRO_Z,
};


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


int main(void) {

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

    struct accelgyro_preprocessor preprocessor;

    accelgyro_preprocessor_init(&preprocessor);
    accelgyro_preprocessor_set_gravity_lowpass(&preprocessor, 0.5f, SAMPLERATE);

    if (!device_is_ready(lsm6dsl_dev)) {
        printk("sensor: device %s not ready.\n", lsm6dsl_dev->name);
        return 0;
    }

    // Setup sensor
    setup_sensor(lsm6dsl_dev);

    // Start high-priority thread for collecting data
    sensor_chunk_reader_start(&reader);

    int iteration = 0;
    float previous_input = 0.0f;
    while (1) {
        const float uptime = k_uptime_get() / 1000.0;

        // check for new data
        const int get_error = k_msgq_get(reader.queue, &chunk, K_NO_WAIT);
        if (get_error == 0) {

            //printk("process-chunk length=%d \n", chunk.length);
            const int run_status = \
                accelgyro_preprocessor_run(&preprocessor, chunk.buffer, chunk.length);

            const float dt = uptime - previous_input;
            printk("features err=%d l=%d dt=%.3f time=%.3f | ", run_status, chunk.length, (double)dt, (double)uptime);
            const int n_features = accelgyro_features_length;
            for (int i=0; i<n_features; i++) {
                printk("%.4f ", (double)preprocessor.features[i]);
            }
            printk("\n");

            const float *gravity = preprocessor.gravity;
            printk("gravity %.2f %.2f %.2f \n",
                (double)gravity[0], (double)gravity[1], (double)gravity[2]);

            // TODO: run through ML model, print outputs
            previous_input = uptime;
        }

#if 0
        printk("main-loop-iter iteration=%d \n", iteration);
#endif

        iteration += 1;
	    k_msleep(100);
    }

    return 0;
}

