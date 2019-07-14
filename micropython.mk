
EMLEARN_MOD_DIR := $(USERMOD_DIR)

SRC_USERMOD += $(EMLEARN_MOD_DIR)/bindings/eml_micropython.c

# so we can include emlearn in bindings
CFLAGS_USERMOD += -I$(EMLEARN_MOD_DIR)/emlearn
