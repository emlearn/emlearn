
#ifndef EMTREES_H
#define EMTREES_H

#include <stdint.h>

typedef float EmtreesValue;

typedef struct _EmtreesNode {
    int8_t feature;
    EmtreesValue value;
    int16_t left;
    int16_t right;
} EmtreesNode;


typedef struct _Emtrees {
    int32_t n_nodes;
    EmtreesNode *nodes;

    int32_t n_trees;
    int32_t *tree_roots;

    // int8_t n_features;
    // int8_t n_classes;
} Emtrees;


typedef enum _EmtreesError {
    EmtreesOK = 0,
    EmtreesUnknownError,
    EmtreesInvalidClassPredicted,
    EmtreesErrorLength,
} EmtreesError;

const char *emtrees_errors[EmtreesErrorLength+1] = {
   "OK",
   "Unknown error",
   "Invalid class predicted",
   "Error length",
};

#ifndef EMTREES_MAX_CLASSES
#define EMTREES_MAX_CLASSES 10
#endif

static EmtreesValue
emtrees_tree_predict(const Emtrees *forest, int32_t tree_root, const EmtreesValue *features, int8_t features_length) {
    int32_t node_idx = tree_root;

    // TODO: see if using a pointer node instead of indirect adressing using node_idx improves perf
    while (forest->nodes[node_idx].feature >= 0) {
        const int8_t feature = forest->nodes[node_idx].feature;
        const EmtreesValue value = features[feature];
        const EmtreesValue point = forest->nodes[node_idx].value;
        //printf("node %d feature %d. %d < %d\n", node_idx, feature, value, point);
        node_idx = (value < point) ? forest->nodes[node_idx].left : forest->nodes[node_idx].right;
    }
    return forest->nodes[node_idx].value;
}

int32_t
emtrees_predict(const Emtrees *forest, const EmtreesValue *features, int8_t features_length) {

    //printf("features %d\n", features_length);
    //printf("trees %d\n", forest->n_trees);
    //printf("nodes %d\n", forest->n_nodes);

    // FIXME: check if number of tree features is bigger than provided
    // FIXME: check if number of classes is bigger than MAX_CLASSES, error
 
    int32_t votes[EMTREES_MAX_CLASSES] = {0};
    for (int32_t i=0; i<forest->n_trees; i++) {
        const int32_t _class = emtrees_tree_predict(forest, forest->tree_roots[i], features, features_length);
        //printf("pred[%d]: %d\n", i, _class);
        if (_class >= 0 && _class < EMTREES_MAX_CLASSES) {
            votes[_class] += 1;
        } else {
            return -EmtreesInvalidClassPredicted;
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


inline float
emtrees_expit(float x) {
    return 1/(1+exp(-x));
}

inline float
emtrees_logit(float p) {
    return log(p/(1-p));
}

// TODO: support emtrees_regress for RandomForestRegressor and ExtraTreesRegressor

int32_t
emtrees_gbm_predict(const Emtrees *model,
        const EmtreesValues *features, int8_t features_length) {

    // FIXME: support multi-class, not just binary  

    // FIXME: pre-compute in the scale factor
    float scale = 0.1;

    // Additive trees
    float score = 0.0;
    for (int32_t i=0; i<forest->n_trees; i++) {
        const float val = emtrees_tree_predict(forest, forest->tree_roots[i],
                                               features, features_length);
        score += scale * val;
    }

    // Apply loss
    // binomial deviation
    // TODO: support other loss types
    const float p = emtrees_expit(score);
    // Decide winner
    return ( p > (1.0f-p)  ) ? 1 : 0;
}

#endif // EMTREES_H
