BIN=led
OBJS=led.o
MCU=atmega328p

CC=avr-gcc
OBJCOPY=avr-objcopy
CFLAGS=-Wall -Os -DF_CPU=16000000UL -mmcu=${MCU}

${BIN}.hex: ${BIN}.elf
	${OBJCOPY} -O ihex -R .eeprom $< $@

${BIN}.elf: ${OBJS}
	${CC} -mmcu=${MCU} -o $@ $^

clean:
	rm -f ${BIN}.elf ${BIN}.hex ${OBJS}

