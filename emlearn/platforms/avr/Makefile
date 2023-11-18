BIN=led
OBJS=led.o
MCU=atmega328p
EXTRA_CFLAGS:=""

CC=avr-gcc
OBJCOPY=avr-objcopy
CFLAGS=-Wall -Os -DF_CPU=16000000UL -mmcu=${MCU} ${EXTRA_CFLAGS}

${BIN}.hex: ${BIN}.elf
	${OBJCOPY} -O ihex -R .eeprom $< $@

${BIN}.elf: ${OBJS}
	${CC} -mmcu=${MCU} -o $@ $^

clean:
	rm -f ${BIN}.elf ${BIN}.hex ${OBJS}

