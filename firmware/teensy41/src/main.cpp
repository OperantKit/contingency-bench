// Teensy 4.1 fake operant chamber for the experiment-io serial backend.
//
// Wire format (experiment-io SerialOperandum / SerialReinforcer):
//   device -> host: {"op":0,"ts":<s>,"kind":"response","seq":<n>}
//   host -> device: {"cmd": "reinforcer", "op": 0, "duration": <s>}
// Extra commands used only by the bring-up host script:
//   {"cmd":"sync","t":<host monotonic s>}  start pressing; ts are offset to the host clock
//   {"cmd":"stop"}                          stop pressing
//   {"cmd":"report"}                        dump press->command latencies (us)
//   {"cmd":"proto","p":"rs","duration":<s>} speak the contingency-rs SerialApparatus protocol
//   {"cmd":"proto","p":"json"}              back to the experiment-io format (the default)
// contingency-rs SerialApparatus format (after proto rs; the control commands above stay JSON):
//   device -> host: R 0
//   host -> device: D <channel> <magnitude>  feeder on for the proto duration
// The report is the same in both formats.
//
// Pins: 2 = feeder (HIGH while the reinforcer is on), 3 = lever-press marker (1 ms pulse).
// The simulated subject presses every kPressPeriodUs, never while the feeder is on.

#include <Arduino.h>

static const uint8_t kFeederPin = 2;
static const uint8_t kPressPin = 3;
static const uint32_t kPressPeriodUs = 300000;
static const uint32_t kPressPulseUs = 1000;
static const size_t kMaxLog = 2000;

static bool active = false;
static double sync_host_s = 0.0;
static uint32_t sync_us = 0;
static uint32_t next_press_us = 0;
static uint32_t press_pulse_start = 0;
static bool press_pulse_on = false;
static uint32_t seq = 0;
static uint32_t last_press_us = 0;
static bool awaiting_cmd = false;

static bool proto_rs = false;
static double rs_duration_s = 0.05;

static bool feeder_on = false;
static uint32_t feeder_start_us = 0;
static uint32_t feeder_len_us = 0;

static uint32_t lat_us[kMaxLog];
static size_t n_lat = 0;

static char line[160];
static size_t line_len = 0;

static double host_ts(uint32_t now_us) {
    return sync_host_s + (double)(uint32_t)(now_us - sync_us) / 1e6;
}

static double field_number(const char* s, const char* key) {
    const char* p = strstr(s, key);
    if (!p) return -1.0;
    p = strchr(p, ':');
    return p ? atof(p + 1) : -1.0;
}

static void start_feeder(double dur, uint32_t now_us) {
    digitalWriteFast(kFeederPin, HIGH);
    feeder_on = true;
    feeder_start_us = now_us;
    feeder_len_us = (uint32_t)(dur * 1e6);
    if (awaiting_cmd && n_lat < kMaxLog) lat_us[n_lat++] = now_us - last_press_us;
    awaiting_cmd = false;
}

static void handle_line(const char* s, uint32_t now_us) {
    if (s[0] == 'D' && s[1] == ' ') {
        start_feeder(rs_duration_s, now_us);
    } else if (strstr(s, "\"reinforcer\"")) {
        double dur = field_number(s, "\"duration\"");
        if (dur < 0) dur = 0.5;
        start_feeder(dur, now_us);
    } else if (strstr(s, "\"proto\"")) {
        proto_rs = strstr(s, "\"rs\"") != nullptr;
        const double dur = field_number(s, "\"duration\"");
        if (dur >= 0) rs_duration_s = dur;
    } else if (strstr(s, "\"sync\"")) {
        sync_host_s = field_number(s, "\"t\"");
        sync_us = now_us;
        active = true;
        n_lat = 0;
        seq = 0;
        next_press_us = now_us + kPressPeriodUs;
    } else if (strstr(s, "\"stop\"")) {
        active = false;
    } else if (strstr(s, "\"report\"")) {
        Serial.printf("{\"kind\":\"report\",\"n\":%u,\"lat_us\":[", (unsigned)n_lat);
        for (size_t i = 0; i < n_lat; i++) Serial.printf(i ? ",%lu" : "%lu", (unsigned long)lat_us[i]);
        Serial.print("]}\n");
    }
}

void setup() {
    pinMode(kFeederPin, OUTPUT);
    pinMode(kPressPin, OUTPUT);
    digitalWriteFast(kFeederPin, LOW);
    digitalWriteFast(kPressPin, LOW);
    Serial.begin(115200);
}

void loop() {
    const uint32_t now = micros();

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            line[line_len] = '\0';
            handle_line(line, micros());
            line_len = 0;
        } else if (line_len < sizeof(line) - 1) {
            line[line_len++] = c;
        }
    }

    // Re-read the clock: handle_line() may have started the feeder after `now` was taken,
    // and `now - feeder_start_us` would wrap to a huge value and end the pulse at once.
    if (feeder_on && (uint32_t)(micros() - feeder_start_us) >= feeder_len_us) {
        digitalWriteFast(kFeederPin, LOW);
        feeder_on = false;
    }
    if (press_pulse_on && (uint32_t)(now - press_pulse_start) >= kPressPulseUs) {
        digitalWriteFast(kPressPin, LOW);
        press_pulse_on = false;
    }

    if (active && !feeder_on && (int32_t)(now - next_press_us) >= 0) {
        const uint32_t t = micros();
        digitalWriteFast(kPressPin, HIGH);
        press_pulse_on = true;
        press_pulse_start = t;
        last_press_us = t;
        awaiting_cmd = true;
        if (proto_rs) {
            Serial.print("R 0\n");
            seq++;
        } else {
            Serial.printf("{\"op\":0,\"ts\":%.6f,\"kind\":\"response\",\"seq\":%lu}\n",
                          host_ts(t), (unsigned long)seq++);
        }
        next_press_us = t + kPressPeriodUs;
    }
}
