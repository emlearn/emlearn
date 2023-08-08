
/**
\brief Nearest Neighbours

k-Nearest-Neighbours (kNN) classification and regression,
as well as outlier detection / anomaly detection / novelty detection

Supports dynamic growth of the model data,
in order to support on-device learning.
*/

#include <eml_common.h>
#include <eml_log.h>
#include <stdint.h>
#include <string.h>

#ifndef EML_NEIGHBORS_LOG_LEVEL
#define EML_NEIGHBORS_LOG_LEVEL 0
#endif

int32_t eml_isqrt(int32_t x)
{
    int32_t q = 1, r = 0;
    while (q <= x) {
        q <<= 2;
    }
    while (q > 1) {
        int32_t t;
        q >>= 2;
        t = x - r - q;
        r >>= 1;
        if (t >= 0) {
            x = t;
            r += q;
        }
    }
    return r;
}

uint32_t
eml_distance_euclidean_int16(const int16_t *a, const int16_t *b, int length)
{
    // FIXME: test this with large values
    uint32_t ret = 0;
    for (int i=0; i<length; i++) {
        const int32_t diff = a[i] - b[i];        
        const int32_t d = diff * diff;
        ret += d;
    }
    ret = eml_isqrt(ret); 
    return ret;
}


typedef struct _EmlNeighborsDistanceItem {
    int16_t index;
    uint32_t distance;
} EmlNeighborsDistanceItem;


static int
eml_neighbors_distance_item_sort_ascending(const void* a, const void* b)
{
    const uint32_t A = ((EmlNeighborsDistanceItem *)a)->distance;
    const uint32_t B = ((EmlNeighborsDistanceItem *)b)->distance;
    if( A == B ) return 0;
    return A < B ? -1 : 1;
}

/* Sort distances while preserving the index */
EmlError
eml_neighbors_sort_distances(EmlNeighborsDistanceItem *distances, size_t length)
{
    qsort(distances, length, sizeof(EmlNeighborsDistanceItem),
        eml_neighbors_distance_item_sort_ascending);
    return EmlOk;
}


typedef struct _EmlNeighborsModel {

    uint16_t n_features;
    int16_t n_items;
    int16_t max_items;

    int16_t *data; // (n_items * n_features)
    int16_t *labels; // n_items

    int16_t k_neighbors;

} EmlNeighborsModel;

EmlError
eml_neighbors_check(EmlNeighborsModel *self,
        int16_t data_length, int16_t labels_length, int16_t distances_length)
{
    const int32_t expect_data_length = self->max_items * self->n_features;
    if (data_length < expect_data_length) {
        return EmlSizeMismatch;
    }
    if (labels_length < self->max_items) {
        return EmlSizeMismatch;
    }
    if (distances_length < self->max_items) {
        return EmlSizeMismatch;
    }
    return EmlOk;
}

EmlError
eml_neighbors_add_item(EmlNeighborsModel *self,
        const int16_t *values, int16_t values_length,
        int16_t label)
{
    EML_PRECONDITION(values_length == self->n_features, EmlSizeMismatch);
    EML_PRECONDITION(self->n_items < self->max_items, EmlSizeMismatch);

    const int index = self->n_items++;
    int16_t *data = self->data + (self->n_features * index);
    memcpy(data, values, sizeof(int16_t)*values_length);
    self->labels[index] = label;    

#if EML_NEIGHBORS_LOG_LEVEL > 2
    EML_LOG_BEGIN("eml_neighbors_add_item");
    EML_LOG_ADD_INTEGER("index", index);
    EML_LOG_ADD_INTEGER("label", label);
    EML_LOG_END();
#endif

    return EmlOk;
}

EmlError
eml_neighbors_infer(EmlNeighborsModel *self,
            const int16_t *features, int features_length,
            EmlNeighborsDistanceItem *distances, int distances_length)
{
    EML_PRECONDITION(distances_length <= self->n_items, EmlSizeMismatch);    
    EML_PRECONDITION(features_length == self->n_features, EmlSizeMismatch);

    // compute distances to all items
    for (int i=0; i<self->n_items; i++) {
        int16_t *item = self->data + (self->n_features * i);
        uint32_t distance = eml_distance_euclidean_int16(features, item, features_length);

        distances[i].index = i;
        distances[i].distance = distance;

#if EML_NEIGHBORS_LOG_LEVEL > 2
        EML_LOG_BEGIN("eml_neighbors_infer_iter");
        EML_LOG_ADD_INTEGER("index", i);
        EML_LOG_ADD_INTEGER("distance", distance);
        EML_LOG_END();
#endif

    }

    return EmlOk;
}

#define EML_NEIGHBORS_MAX_CLASSES 10


EmlError
eml_neighbors_find_nearest(EmlNeighborsModel *self,
        EmlNeighborsDistanceItem *distances, int distances_length,
        int k, int16_t *out)
{
    EML_PRECONDITION(k <= distances_length, EmlSizeMismatch);

    // argsort by distance. NOTE: sorts in-place
    eml_neighbors_sort_distances(distances, distances_length);

    // FIXME: avoid hardcoding length
    static int16_t votes[EML_NEIGHBORS_MAX_CLASSES];
    memset(votes, 0, sizeof(int16_t) * EML_NEIGHBORS_MAX_CLASSES);

    // merge predictions for top-k matches
    for (int i=0; i<k; i++) {
        EmlNeighborsDistanceItem d = distances[i];
        const int16_t label = self->labels[d.index];         
        if (label < 0 || label >= EML_NEIGHBORS_MAX_CLASSES) {
            return EmlUnknownError;
        }
        votes[label] += 1;

#if EML_NEIGHBORS_LOG_LEVEL > 2
        EML_LOG_BEGIN("eml_neighbors_find_nearest_k_iter");
        EML_LOG_ADD_INTEGER("index", i);
        EML_LOG_ADD_INTEGER("distance", d.distance);
        EML_LOG_ADD_INTEGER("label", label);
        EML_LOG_END();
#endif
    }

    int32_t most_voted_class = -1;
    int32_t most_voted_votes = 0;
    for (int32_t i=0; i<EML_NEIGHBORS_MAX_CLASSES; i++) {

#if EML_NEIGHBORS_LOG_LEVEL > 1
        EML_LOG_BEGIN("eml_neighbors_find_nearest_votes_iter");
        EML_LOG_ADD_INTEGER("index", i);
        EML_LOG_ADD_INTEGER("votes", votes[i]);
        EML_LOG_END();
#endif

        if (votes[i] > most_voted_votes) {
            most_voted_class = i;
            most_voted_votes = votes[i];
        }
    }
    *out = most_voted_class;

    return EmlOk;
}

EmlError
eml_neighbors_predict(EmlNeighborsModel *self,
        const int16_t *features, int features_length,
        EmlNeighborsDistanceItem *distances, int distances_length,
        int16_t *out)
{
    // NOTE: Preconditions checked inside _infer()
    // Compute distances
    const EmlError infer_err = \
        eml_neighbors_infer(self, features, features_length, distances, distances_length);
    if (infer_err != EmlOk) {
        return infer_err;
    }

    // Find kNN predictions 
    int16_t label = -1;
    EmlError find_err = \
        eml_neighbors_find_nearest(self, distances, distances_length, self->k_neighbors, &label);
    if (find_err != EmlOk) {
        return find_err;
    }
    *out = label;

    return EmlOk;
}

