
EMLEARN_INCLUDE_DIR := /home/jon/projects/emlearn/emlearn

#CFLAGS := -g 
CFLAGS := -g -fsanitize=address,undefined,leak -Wall -Werror -Wno-error=unused-function
CFLAGS += -I ${EMLEARN_INCLUDE_DIR} -I./ -I./build -lm

all: check

build/gravity_filter.h: tools/gravity_lowpass.py
	mkdir -p build/
	python tools/gravity_lowpass.py --out $@ --format header

build/motion_model.h: combined.estimator.pickle
	mkdir -p build/
	python tools/convert_model.py --out $@ --model $<

test_processing: build/gravity_filter.h
	gcc -o ./build/motion_preprocess tools/motion_process.c  ${CFLAGS}
	./build/motion_preprocess --input sensordata.csv --output build/features.csv

test_model: build/motion_model.h build/gravity_filter.h
	gcc -o build/motion_process_model tools/motion_process.c ${CFLAGS} -DMOTION_MODEL_FILE=\"motion_model.h\"
	build/motion_process_model --input sensordata.csv --output build/output.csv

check: test_processing test_model


.PHONY: test_processing

