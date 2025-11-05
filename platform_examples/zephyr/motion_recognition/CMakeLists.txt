# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(lsm6dsl)

FILE(GLOB app_sources src/*.c)
target_sources(app PRIVATE ${app_sources})

set(MOTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/motion_recognition/)
set(MOTION_MODEL_IN ${CMAKE_CURRENT_SOURCE_DIR}/combined.estimator.pickle)
set(MOTION_MODEL_OUT ${CMAKE_CURRENT_BINARY_DIR}/motion_model.h)

set(GRAVITY_FILTER_OUT ${CMAKE_CURRENT_BINARY_DIR}/motion_gravity_lowpass.h)

# Custom command for motion_model.h
add_custom_command(
  OUTPUT ${MOTION_MODEL_OUT}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND python3 ${MOTION_DIR}/tools/convert_model.py --out ${MOTION_MODEL_OUT} --model ${MOTION_MODEL_IN}
  DEPENDS ${MOTION_DIR}/tools/convert_model.py
  COMMENT "Generating motion_model.h"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(generate_model ALL
  DEPENDS ${MOTION_MODEL_IN} ${MOTION_MODEL_OUT}
)

# Custom command for gravity filter
add_custom_command(
  OUTPUT ${GRAVITY_FILTER_OUT}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND python3 ${MOTION_DIR}/tools/gravity_lowpass.py --out ${GRAVITY_FILTER_OUT} --format header
  DEPENDS ${MOTION_DIR}/tools/gravity_lowpass.py
  COMMENT "Generating gravity_lowpass.h"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(generate_filter ALL
  DEPENDS ${GRAVITY_FILTER_OUT}
)

target_include_directories(app PRIVATE 
    ${MOTION_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ZEPHYR_BASE}/samples/subsys/usb/common
)

add_dependencies(app generate_model generate_filter)

target_sources(app PRIVATE
    # ${ZEPHYR_BASE}/samples/subsys/usb/common/sample_usbd_init.c
)
