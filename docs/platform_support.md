
## Platform support

emlearn works on *any* hardware platform and SDK with **C99 compiler** support.
That is practically *all* microcontroller, embedded device and most DSP platforms from the last 20 years.
For example, emlearn is known to work together with the following.

### Software frameworks and RTOS

- Zephyr. Has [first-party support](https://emlearn.readthedocs.io/en/latest/getting_started_zephyr.html).
- Arduino. Has [first-party support](https://emlearn.readthedocs.io/en/latest/getting_started_arduino.html).
- STM32Cube
- ESP-IDF
- FreeRTOS
- Contiki-NG
- [RIOT OS](https://www.riot-os.org/) has a [package for emlearn.](https://github.com/RIOT-OS/RIOT/tree/master/tests/pkg/emlearn).
- Apache mynewt
- Azure RTOS ThreadX
- and so on..

### Microcontroller hardware platforms

- ST STM32F4/STM32L4/STM32F1 et.c.
- Espressif Xtensa ESP8266/ESP32/ESP32-S3 and RISC-V ESP32-C5/C3 etc.
- Nordic NRF51/NRF52/NRF53/NRF54/NRF91
- Atmel AVR8/AVR32
- Raspberry PI Pico / RP2040 / RP2350
- and so on...

### Embedded OS platforms

- Linux
- Windows 10 IoT
- Android
- and so on...

### Other programming languages

Since emlearn is a standard C library, it works well with any language that supports C bindings.

- C++
- Rust
- Zig
- Nim
- MicroPython. Via [emlearn-micropython](https://emlearn-micropython.readthedocs.io/en/latest/).
- Python. Using CFFI or pybind11
- Node.js/JavaScript
- WebAssembly/WASM. Using Emscripten. [Quickstart](https://emlearn.readthedocs.io/en/latest/getting_started_browser.html).
- Java. Using JNI/JNA


### More

This list is **not exhaustive**, and emlearn should work on *any platform* with a C99 compiler.
If you have used emlearn on a platform not mentioned here, please let us know.

