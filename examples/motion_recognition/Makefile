
EMLEARN_INCLUDE_DIR := /home/jon/projects/emlearn/emlearn

#CFLAGS := -g 
CFLAGS := -g -fsanitize=address,undefined,leak -Wall -Werror -Wno-error=unused-function

all: check

build/gravity_filter.h:
	mkdir -p build/
	python tools/gravity_lowpass.py

build/motion_model.h:
	mkdir -p build/
	python tools/convert_model.py

test_processing: build/gravity_filter.h
	gcc -o ./build/motion_preprocess tools/motion_process.c -I ${EMLEARN_INCLUDE_DIR} -I./ -I./build -lm ${CFLAGS}
	./build/motion_preprocess sensordata.csv build/features.csv 50

test_model: build/motion_model.h build/gravity_filter.h
	gcc -o build/motion_process_model tools/motion_process.c -I ${EMLEARN_INCLUDE_DIR} -I./ -I./build -lm ${CFLAGS} -DMOTION_MODEL_FILE=\"motion_model.h\"
	build/motion_process_model sensordata.csv build/output.csv 50

check: test_processing test_model


.PHONY: test_processing

