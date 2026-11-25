// Runs setup() and loop() of src/main.cpp on the host. Standard input is what
// the device receives over serial, standard output is what it sends.
//
// Real clock (the default): micros() is the monotonic clock since the start.
// The program ends when standard input ends.
//
// Virtual clock (--virtual-clock): micros() advances by a fixed step after each
// loop() and stands still inside it. All of standard input is read before the
// clock first advances, so the same input gives the same output however it
// arrives; then loop() runs --loops-after-eof more times.

#include "Arduino.h"

#include <errno.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <thread>

static const uint64_t kDefaultStepUs = 100;
// The firmware ticks every 1000 us; a longer step would skip ticks.
static const uint64_t kMaxStepUs = 1000;
static const std::chrono::microseconds kIdleSleep(100);

struct Options {
    bool virtual_clock = false;
    uint64_t step_us = kDefaultStepUs;
    uint64_t loops_after_eof = 0;
    bool log_pins = false;
};

static Options options;
static uint64_t virtual_us = 0;

static unsigned char rx[4096];
static size_t rx_len = 0;
static size_t rx_pos = 0;
static bool rx_eof = false;

HostSerial Serial;

uint32_t micros() {
    if (options.virtual_clock) return (uint32_t)virtual_us;
    using namespace std::chrono;
    static const steady_clock::time_point t0 = steady_clock::now();
    return (uint32_t)duration_cast<microseconds>(steady_clock::now() - t0).count();
}

void pinMode(uint8_t, uint8_t) {}

void digitalWriteFast(uint8_t pin, uint8_t value) {
    if (!options.log_pins) return;
    fprintf(stderr, "{\"pin\":%u,\"value\":%u,\"us\":%lu}\n", (unsigned)pin, (unsigned)value,
            (unsigned long)micros());
}

void HostSerial::begin(long) { setvbuf(stdout, nullptr, _IOLBF, 0); }

// Reads ahead, so that read() never has to return -1 after available() said 1:
// main.cpp would take that -1 as a received character.
int HostSerial::available() {
    if (rx_pos < rx_len) return 1;
    if (rx_eof) return 0;
    struct pollfd p = {STDIN_FILENO, POLLIN, 0};
    if (poll(&p, 1, options.virtual_clock ? -1 : 0) <= 0) return 0;
    const ssize_t n = ::read(STDIN_FILENO, rx, sizeof(rx));
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) return 0;
    rx_pos = 0;
    rx_len = n > 0 ? (size_t)n : 0;
    rx_eof = n <= 0;
    return rx_len > 0 ? 1 : 0;
}

int HostSerial::read() { return available() ? rx[rx_pos++] : -1; }

size_t HostSerial::print(const char* s) { return fwrite(s, 1, strlen(s), stdout); }

int HostSerial::printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int n = vprintf(fmt, args);
    va_end(args);
    return n;
}

static int usage(const char* program) {
    fprintf(stderr,
            "usage: %s [--virtual-clock [--step-us N] [--loops-after-eof N]] [--pins]\n"
            "  --virtual-clock      micros() advances by a fixed step per loop()\n"
            "  --step-us N          the step, 1 to %lu (default %lu)\n"
            "  --loops-after-eof N  loop() calls after the end of standard input (default 0)\n"
            "  --pins               write each pin change to standard error\n",
            program, (unsigned long)kMaxStepUs, (unsigned long)kDefaultStepUs);
    return 2;
}

static bool parse_count(const char* s, uint64_t* out) {
    if (!s || *s < '0' || *s > '9') return false;
    char* end = nullptr;
    errno = 0;
    *out = strtoull(s, &end, 10);
    return errno == 0 && *end == '\0';
}

static bool parse_options(int argc, char** argv) {
    bool virtual_only = false;
    for (int i = 1; i < argc; i++) {
        const char* const arg = argv[i];
        if (strcmp(arg, "--virtual-clock") == 0) {
            options.virtual_clock = true;
        } else if (strcmp(arg, "--pins") == 0) {
            options.log_pins = true;
        } else if (strcmp(arg, "--step-us") == 0) {
            if (!parse_count(argv[++i], &options.step_us)) return false;
            if (options.step_us < 1 || options.step_us > kMaxStepUs) return false;
            virtual_only = true;
        } else if (strcmp(arg, "--loops-after-eof") == 0) {
            if (!parse_count(argv[++i], &options.loops_after_eof)) return false;
            virtual_only = true;
        } else {
            return false;
        }
    }
    return options.virtual_clock || !virtual_only;
}

int main(int argc, char** argv) {
    if (!parse_options(argc, argv)) return usage(argv[0]);
    setup();
    for (uint64_t after_eof = 0;;) {
        loop();
        if (rx_eof && after_eof++ >= options.loops_after_eof) break;
        if (options.virtual_clock) {
            virtual_us += options.step_us;
        } else {
            std::this_thread::sleep_for(kIdleSleep);
        }
    }
    return fflush(stdout) == 0 ? 0 : 1;
}
