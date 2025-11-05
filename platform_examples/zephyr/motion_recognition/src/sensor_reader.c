
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>

#include "sensor_reader.h"


// Designed to be called in a new thread
static void
sensor_chunk_reader_task(void *context, void *, void *)
{
    struct sensor_chunk_reader * self = (struct sensor_chunk_reader *)context;
    float values[SENSOR_CHUNK_READER_MAX_CHANNELS];

    // MAYBE: support a way of exiting loop gracefully?
    while (1) {

        // Read data
	    const int fetch_ret = sensor_sample_fetch(self->dev);
	    if (fetch_ret < 0) {
            self->fetch_errors += 0;

	    }
	    for (size_t i = 0; i < self->n_channels; i++) {
            struct sensor_value value;
		    const int get_ret = sensor_channel_get(self->dev, self->channels[i], &value);
            values[i] = sensor_value_to_double(&value);

		    if (get_ret < 0) {
                self->get_errors += 1;
		    }
	    }

        // Buffer received data
        const int read_offset = (self->read_samples_index*self->n_channels);
        memcpy(self->read_samples+read_offset, values, sizeof(float)*self->n_channels);
        self->read_samples_index += 1;

#if 0
        printk("reader-task-got-data index=%d \n", self->read_samples_index);
#endif

        // TODO: respect hop_length, using overlap
        if (self->read_samples_index == self->window_length) {
            const int out_length = self->n_channels * self->window_length;

            // Push onto output buffer
            memcpy(self->output_buffer, self->read_samples, sizeof(float)*out_length);
            self->read_samples_index = 0;

            // Send as message
            struct sensor_chunk_msg msg = { 1, self->output_buffer, out_length };
#if 0
    printk("reader-task-got-data index=%d \n", self->read_samples_index);
#endif
            const int put_status = k_msgq_put(self->queue, &msg, K_NO_WAIT);
            if (put_status < 0) {
                self->put_errors += 1;
            }

        }

        const int wait_timeout_us = 1000000/self->samplerate;
        k_usleep(wait_timeout_us);
    }
}


void
sensor_chunk_reader_start(struct sensor_chunk_reader *self) 
{
    // start new thread
    k_tid_t thread_id = k_thread_create(self->thread, self->stack,
                                     self->stack_size,
                                     sensor_chunk_reader_task,
                                     self, NULL, NULL,
                                     self->thread_priority, 0, K_NO_WAIT);

    self->thread_id = thread_id;
}


void
sensor_chunk_reader_stop(struct sensor_chunk_reader *self) 
{
    // FIXME: should probably do something more graceful
    k_thread_abort(self->thread_id);
}

