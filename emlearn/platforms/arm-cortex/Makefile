BIN=led
OBJS=led.o

# Specify the compiler options specific to the particular microcontroller series
# Ref https://github.com/ARM-software/toolchain-gnu-bare-metal/blob/master/readme.txt
ARM_CORTEX_CFLAGS:=-mthumb -mcpu=cortex-m0plus

# Extra compiler flags, like includes etc
EXTRA_CFLAGS:=""

CC=arm-none-eabi-gcc
OBJCOPY=arm-none-eabi-objcopy
CFLAGS=-Wall -Os ${ARM_CORTEX_CFLAGS} ${EXTRA_CFLAGS} 

${BIN}.hex: ${BIN}.elf
	${OBJCOPY} -O ihex -R .eeprom $< $@

${BIN}.elf: ${OBJS}
	${CC} --specs=nosys.specs ${ARM_CORTEX_CFLAGS} -o $@ $^

clean:
	rm -f ${BIN}.elf ${BIN}.hex ${OBJS}

