# Copyright (c) 2024 emlearn
# SPDX-License-Identifier: MIT

if(CONFIG_EMLEARN)

  set(EMLEARN_MODULE_DIR ${ZEPHYR_CURRENT_MODULE_DIR})

  zephyr_library()

  zephyr_include_directories(
    ${EMLEARN_MODULE_DIR}/emlearn
  )

endif()
