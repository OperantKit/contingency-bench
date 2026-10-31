// Teensy 4.1 fake operant chamber for the experiment-io serial backend.
//
// Host mode (the host decides; the default):
//   device -> host: {"op":0,"ts":<s>,"kind":"response","seq":<n>}
//   host -> device: {"cmd": "reinforcer", "op": 0, "duration": <s>}
// Extra commands used only by the bring-up host script:
//   {"cmd":"sync","t":<host monotonic s>}  start pressing; ts are offset to the host clock
//   {"cmd":"stop"}                          stop pressing
//   {"cmd":"report"}                        dump latencies (us); see below
//   {"cmd":"proto","p":"rs","duration":<s>} speak the contingency-rs SerialApparatus protocol
//   {"cmd":"proto","p":"json"}              back to the experiment-io format (the default)
// contingency-rs SerialApparatus format (after proto rs; the control commands above stay JSON):
//   device -> host: R 0
//   host -> device: D <channel> <magnitude>  feeder on for the proto duration
//
// Device mode (the device decides; only CRF and FR):
//   {"cmd":"load","schedule":{"type":"FR","params":{"n":5}},
//    "reinforcer":{"op":0,"duration":<s>},"stop_after":{"reinforcers":<n>,"seconds":<s>}}
//       -> {"kind":"loaded",...} or {"kind":"error","code":"unsupported_schedule"|"bad_params",...}
//   {"cmd":"probe","now":<s>,"event":null|{"operandum":"main",...}}  one step, no real time,
//       only before start  -> {"kind":"outcome","reinforced":...,"reinforcer":{...}}
//   {"cmd":"start"}  -> {"kind":"started","ts":<s>}; pressing begins
//   {"cmd":"stop"}   -> {"kind":"stopped","ts":<s>,"n_responses":<n>,"n_reinforcers":<n>}
//   events: response {op,ts,seq,reinforced}, reinforcer_start {op,ts,seq},
//           reinforcer_end {op,ts,seq}, stopped (also when stop_after is met)
//   ts is the device clock in seconds (micros() extended to 64 bits, never reset);
//   seq goes up by 1 for every event line. "reinforcer" is refused with an error.
//
// report: {"kind":"report","n":<n>,"lat_us":[...]}. Host mode: press -> reinforcer command
// read. Device mode: press -> feeder on (the decision inside the Teensy).
//
// Pins: 2 = feeder (HIGH while the reinforcer is on), 3 = lever-press marker (1 ms pulse).
// The simulated subject presses every kPressPeriodUs, never while the feeder is on.

#include <Arduino.h>

static const uint8_t kFeederPin = 2;
static const uint8_t kPressPin = 3;
static const uint32_t kPressPeriodUs = 300000;
static const uint32_t kPressPulseUs = 1000;
static const size_t kMaxLog = 2000;
static const char* kFirmwareId = "contingency-bench-teensy41-crf-fr";

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

static char line[512];
static size_t line_len = 0;

// Device mode state.
static bool loaded = false;
static bool device_mode = false;  // a loaded schedule is running (between start and stopped)
static uint32_t ratio_n = 1;
static uint32_t ratio_count = 0;
static uint32_t probe_count = 0;
static double dev_duration_s = 0.05;
static long stop_reinforcers = -1;
static double stop_seconds = -1.0;
static double start_ts = 0.0;
static uint32_t n_responses = 0;
static uint32_t n_reinforcers = 0;

static uint32_t clock_hi = 0;
static uint32_t clock_last = 0;

static double device_s(uint32_t now_us) {
    if (now_us < clock_last) clock_hi++;
    clock_last = now_us;
    return ((double)clock_hi * 4294967296.0 + (double)now_us) / 1e6;
}

static double host_ts(uint32_t now_us) {
    return sync_host_s + (double)(uint32_t)(now_us - sync_us) / 1e6;
}

// Pointer to the value after "key": (spaces skipped), or nullptr.
static const char* value_of(const char* s, const char* key) {
    const char* p = strstr(s, key);
    if (!p) return nullptr;
    p = strchr(p + strlen(key), ':');
    if (!p) return nullptr;
    p++;
    while (*p == ' ') p++;
    return p;
}

static double field_number(const char* s, const char* key) {
    const char* p = value_of(s, key);
    return p ? atof(p) : -1.0;
}

static bool cmd_is(const char* s, const char* name) {
    const char* p = value_of(s, "\"cmd\"");
    if (!p || *p != '"') return false;
    const size_t n = strlen(name);
    return strncmp(p + 1, name, n) == 0 && p[1 + n] == '"';
}

static void emit_error(const char* code, const char* msg) {
    Serial.printf("{\"kind\":\"error\",\"code\":\"%s\",\"msg\":\"%s\",\"seq\":%lu}\n", code, msg,
                  (unsigned long)seq++);
}

static void emit_stopped(uint32_t now_us) {
    Serial.printf("{\"kind\":\"stopped\",\"ts\":%.6f,\"seq\":%lu,\"n_responses\":%lu,"
                  "\"n_reinforcers\":%lu}\n",
                  device_s(now_us), (unsigned long)seq++, (unsigned long)n_responses,
                  (unsigned long)n_reinforcers);
}

static void start_feeder(double dur, uint32_t now_us) {
    digitalWriteFast(kFeederPin, HIGH);
    feeder_on = true;
    feeder_start_us = now_us;
    feeder_len_us = (uint32_t)(dur * 1e6);
    if (awaiting_cmd && n_lat < kMaxLog) lat_us[n_lat++] = now_us - last_press_us;
    awaiting_cmd = false;
}

static void handle_load(const char* s) {
    loaded = false;
    const char* t = value_of(s, "\"type\"");
    if (!t || *t != '"') {
        emit_error("bad_params", "schedule.type missing");
        return;
    }
    if (strncmp(t, "\"CRF\"", 5) == 0) {
        ratio_n = 1;
    } else if (strncmp(t, "\"FR\"", 4) == 0) {
        const double n = field_number(s, "\"n\"");
        if (n < 1.0 || n != (double)(uint32_t)n) {
            emit_error("bad_params", "FR needs an integer params.n >= 1");
            return;
        }
        ratio_n = (uint32_t)n;
    } else {
        emit_error("unsupported_schedule", "this firmware decides CRF and FR only");
        return;
    }
    const double dur = field_number(s, "\"duration\"");
    dev_duration_s = dur >= 0 ? dur : 0.05;
    const double r = field_number(s, "\"reinforcers\"");
    stop_reinforcers = r >= 1.0 ? (long)r : -1;
    stop_seconds = field_number(s, "\"seconds\"");
    ratio_count = 0;
    probe_count = 0;
    loaded = true;
    Serial.printf("{\"kind\":\"loaded\",\"schedule\":{\"type\":\"%s\",\"n\":%lu},"
                  "\"firmware\":\"%s\",\"supported\":[\"CRF\",\"FR\"]}\n",
                  ratio_n == 1 && strncmp(t, "\"CRF\"", 5) == 0 ? "CRF" : "FR",
                  (unsigned long)ratio_n, kFirmwareId);
}

static void handle_probe(const char* s) {
    if (!loaded || device_mode) {
        emit_error("bad_state", "probe needs a loaded schedule and no running session");
        return;
    }
    const double now = field_number(s, "\"now\"");
    const char* ev = value_of(s, "\"event\"");
    bool reinforced = false;
    if (ev && *ev == '{') {
        probe_count++;
        if (probe_count >= ratio_n) {
            probe_count = 0;
            reinforced = true;
        }
    }
    if (reinforced) {
        Serial.printf("{\"kind\":\"outcome\",\"reinforced\":true,\"reinforcer\":"
                      "{\"time\":%.6f,\"magnitude\":1.0,\"label\":\"SR+\"}}\n",
                      now);
    } else {
        Serial.print("{\"kind\":\"outcome\",\"reinforced\":false}\n");
    }
}

static void handle_start(uint32_t now_us) {
    if (!loaded) {
        emit_error("bad_state", "start needs a loaded schedule");
        return;
    }
    device_mode = true;
    active = true;
    proto_rs = false;
    n_lat = 0;
    seq = 0;
    n_responses = 0;
    n_reinforcers = 0;
    ratio_count = 0;
    awaiting_cmd = false;
    start_ts = device_s(now_us);
    Serial.printf("{\"kind\":\"started\",\"ts\":%.6f,\"seq\":%lu}\n", start_ts, (unsigned long)seq++);
    next_press_us = now_us + kPressPeriodUs;
}

static void finish_device(uint32_t now_us) {
    active = false;
    device_mode = false;
    loaded = false;
    emit_stopped(now_us);
}

static void handle_line(const char* s, uint32_t now_us) {
    if (cmd_is(s, "load")) {
        handle_load(s);
    } else if (cmd_is(s, "probe")) {
        handle_probe(s);
    } else if (cmd_is(s, "start")) {
        handle_start(now_us);
    } else if (device_mode && cmd_is(s, "stop")) {
        finish_device(now_us);
    } else if (device_mode && (cmd_is(s, "reinforcer") || (s[0] == 'D' && s[1] == ' '))) {
        emit_error("bad_state", "the device decides in device mode; reinforcer refused");
    } else if (s[0] == 'D' && s[1] == ' ') {
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
        device_mode = false;
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

// A press in device mode: decide, turn the feeder on first, then report the events.
static void device_press(uint32_t t) {
    n_responses++;
    ratio_count++;
    const bool reinforced = ratio_count >= ratio_n;
    if (reinforced) {
        ratio_count = 0;
        start_feeder(dev_duration_s, micros());
        n_reinforcers++;
    }
    const uint32_t on_us = feeder_start_us;
    Serial.printf("{\"kind\":\"response\",\"op\":0,\"ts\":%.6f,\"seq\":%lu,\"reinforced\":%s}\n",
                  device_s(t), (unsigned long)seq++, reinforced ? "true" : "false");
    if (reinforced) {
        Serial.printf("{\"kind\":\"reinforcer_start\",\"op\":0,\"ts\":%.6f,\"seq\":%lu}\n",
                      device_s(on_us), (unsigned long)seq++);
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
    device_s(now);

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
    const uint32_t t_off = micros();
    if (feeder_on && (uint32_t)(t_off - feeder_start_us) >= feeder_len_us) {
        digitalWriteFast(kFeederPin, LOW);
        feeder_on = false;
        if (device_mode) {
            Serial.printf("{\"kind\":\"reinforcer_end\",\"op\":0,\"ts\":%.6f,\"seq\":%lu}\n",
                          device_s(t_off), (unsigned long)seq++);
            if (stop_reinforcers > 0 && (long)n_reinforcers >= stop_reinforcers) {
                finish_device(t_off);
            }
        }
    }
    if (press_pulse_on && (uint32_t)(now - press_pulse_start) >= kPressPulseUs) {
        digitalWriteFast(kPressPin, LOW);
        press_pulse_on = false;
    }
    if (device_mode && stop_seconds > 0 && device_s(micros()) - start_ts >= stop_seconds &&
        !feeder_on) {
        finish_device(micros());
    }

    if (active && !feeder_on && (int32_t)(now - next_press_us) >= 0) {
        const uint32_t t = micros();
        digitalWriteFast(kPressPin, HIGH);
        press_pulse_on = true;
        press_pulse_start = t;
        last_press_us = t;
        awaiting_cmd = true;
        if (device_mode) {
            device_press(t);
        } else if (proto_rs) {
            Serial.print("R 0\n");
            seq++;
        } else {
            Serial.printf("{\"op\":0,\"ts\":%.6f,\"kind\":\"response\",\"seq\":%lu}\n",
                          host_ts(t), (unsigned long)seq++);
        }
        next_press_us = t + kPressPeriodUs;
    }
}
