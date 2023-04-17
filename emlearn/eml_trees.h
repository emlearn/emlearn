
#ifndef EML_TREES_H
#define EML_TREES_H

#include <stdint.h>
#include <math.h>
#include <eml_common.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _EmlTreesNode {
    int8_t feature;
    float value;
    int16_t left;
    int16_t right;
} EmlTreesNode;


typedef struct _EmlTrees {
    int32_t n_nodes;
    EmlTreesNode *nodes;

    int32_t n_trees;
    int32_t *tree_roots;

    // int8_t n_features;
    // int8_t n_classes;
} EmlTrees;

typedef enum _EmlTreesError {
    EmlTreesOK = 0,
    EmlTreesUnknownError,
    EmlTreesInvalidClassPredicted,
    EmlTreesErrorLength,
} EmlTreesError;

const char *eml_trees_errors[EmlTreesErrorLength+1] = {
   "OK",
   "Unknown error",
   "Invalid class predicted",
   "Error length",
};

#ifndef EMTREES_MAX_CLASSES
#define EMTREES_MAX_CLASSES 10
#endif

static int32_t
eml_trees_predict_tree(const EmlTrees *forest, int32_t tree_root, const float *features, int8_t features_length) {
    int32_t node_idx = tree_root;

    // TODO: see if using a pointer node instead of indirect adressing using node_idx improves perf
    while (forest->nodes[node_idx].feature >= 0) {
        const int8_t feature = forest->nodes[node_idx].feature;
        const float value = features[feature];
        const float point = forest->nodes[node_idx].value;
        //printf("node %d feature %d. %d < %d\n", node_idx, feature, value, point);
        node_idx = (value < point) ? forest->nodes[node_idx].left : forest->nodes[node_idx].right;
    }
    return forest->nodes[node_idx].value;
}

int32_t
eml_trees_predict(const EmlTrees *forest, const float *features, int8_t features_length) {

    //printf("features %d\n", features_length);
    //printf("trees %d\n", forest->n_trees);
    //printf("nodes %d\n", forest->n_nodes);

    // FIXME: check if number of tree features is bigger than provided
    // FIXME: check if number of classes is bigger than MAX_CLASSES, error
 
    int32_t votes[EMTREES_MAX_CLASSES] = {0};
    for (int32_t i=0; i<forest->n_trees; i++) {
        const int32_t _class = eml_trees_predict_tree(forest, forest->tree_roots[i], features, features_length);
        //printf("pred[%d]: %d\n", i, _class);
        if (_class >= 0 && _class < EMTREES_MAX_CLASSES) {
            votes[_class] += 1;
        } else {
            return -EmlTreesInvalidClassPredicted;
        }
    }
    
    int32_t most_voted_class = -1;
    int32_t most_voted_votes = 0;
    for (int32_t i=0; i<EMTREES_MAX_CLASSES; i++) {
        //printf("votes[%d]: %d\n", i, votes[i]);
        if (votes[i] > most_voted_votes) {
            most_voted_class = i;
            most_voted_votes = votes[i];
        }
    }

    return most_voted_class;
}


EmlError
eml_trees_regress(const EmlTrees *forest,
        const float *features, int8_t features_length,
        float *out, int8_t out_length)
{
    if (out_length < 1) {
        return EmlSizeMismatch;
    }

    float sum = 0; 

    for (int32_t i=0; i<forest->n_trees; i++) {
        const float val = eml_trees_predict_tree(forest, forest->tree_roots[i], features, features_length);
        sum += val;
    }

    out[0] = sum / forest->n_trees;

    return EmlOk;
}

float
eml_trees_regress1(const EmlTrees *forest,
        const float *features, int8_t features_length)
{
    float out[1];
    EmlError err = eml_trees_regress(forest,
        features, features_length,
        out, 1);
    if (err != EmlOk) {    
        return NAN;
    }
    return out[0];
}

#ifdef __cplusplus
}
#endif

#endif // EML_TREES_H
