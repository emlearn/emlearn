
#ifndef SENSOR_CHUNK_READER_MAX_CHANNELS
#define SENSOR_CHUNK_READER_MAX_CHANNELS 9
#endif

struct sensor_chunk_reader {
    int window_length;
    int hop_length;
    int samplerate;

    float *read_samples;
    int read_samples_index;

    float *output_buffer;
    int output_buffer_index;

    int n_channels;
    enum sensor_channel *channels;

    const struct device *dev;
    struct k_msgq *queue;
    struct k_thread *thread;
    int thread_priority;
    k_tid_t thread_id;
    k_thread_stack_t *stack;
    size_t stack_size;

    int fetch_errors;
    int get_errors;
    int put_errors;
};

struct sensor_chunk_msg {
    int sample_no;
    float *buffer; // n_channels * window_length
    size_t length;
};

void sensor_chunk_reader_start(struct sensor_chunk_reader *self);
void sensor_chunk_reader_stop(struct sensor_chunk_reader *self);

