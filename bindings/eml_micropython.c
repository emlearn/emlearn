// Bindings for MicroPython

// Inlcude emlearn
#include <eml_trees.h>

// Include required definitions first.
#include "py/obj.h"
#include "py/runtime.h"
#include "py/builtin.h"


STATIC mp_obj_t emlearn_add_ints(mp_obj_t a_obj, mp_obj_t b_obj) {
    // Extract the ints from the micropython input objects
    int a = mp_obj_get_int(a_obj);
    int b = mp_obj_get_int(b_obj);

    // Calculate the addition and convert to MicroPython object.
    return mp_obj_new_int(a + b);
}
// Define a Python reference to the function above
STATIC MP_DEFINE_CONST_FUN_OBJ_2(emlearn_add_ints_obj, emlearn_add_ints);


// Trees

STATIC mp_obj_t emlearn_mp_trees_load(mp_obj_t in_obj) {
    // Extract the ints from the micropython input objects
    size_t len;
    const char *data = mp_obj_str_get_data(in_obj, &len);

    fprintf(stderr, "emlearn_mp_load: ");
    for (size_t i =0; i<len; i++) {
        fprintf(stderr, "%c", data[i] );
    }
    fprintf(stderr, "\n");

    // Calculate the addition and convert to MicroPython object.
    return mp_obj_new_int(len);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(emlearn_mp_trees_load_obj, emlearn_mp_trees_load);

// Trees end


// Module properties
// All identifiers and strings are written as MP_QSTR_xxx and will be
// optimized to word-sized integers by the build system (interned strings).
STATIC const mp_rom_map_elem_t emlearn_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_emlearn) },
    { MP_ROM_QSTR(MP_QSTR_add_ints), MP_ROM_PTR(&emlearn_add_ints_obj) },
    { MP_ROM_QSTR(MP_QSTR_trees_load), MP_ROM_PTR(&emlearn_mp_trees_load_obj) },
};
STATIC MP_DEFINE_CONST_DICT(emlearn_module_globals, emlearn_module_globals_table);

// Define module object.
const mp_obj_module_t emlearn_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&emlearn_module_globals,
};

// Register the module to make it available in Python
MP_REGISTER_MODULE(MP_QSTR_emlearn, emlearn_user_cmodule, MODULE_EMLEARN_ENABLED);
