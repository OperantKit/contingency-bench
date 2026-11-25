// Host stand-in for the part of the Arduino API that src/main.cpp uses.
// Only the host build (build.sh) puts this directory on the include path.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HIGH 1
#define LOW 0
#define OUTPUT 1

uint32_t micros();
void pinMode(uint8_t pin, uint8_t mode);
void digitalWriteFast(uint8_t pin, uint8_t value);

// The serial port: reads standard input, writes standard output.
struct HostSerial {
    void begin(long baud);
    int available();
    int read();
    size_t print(const char* s);
    int printf(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
};
extern HostSerial Serial;

void setup();
void loop();
