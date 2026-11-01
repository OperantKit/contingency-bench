/* Schedule decisions from contingency-rs (rust/src/lib.rs). */
#ifndef BENCH_DECIDE_H
#define BENCH_DECIDE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  BD_OK = 0,
  BD_UNSUPPORTED = 1,  /* error code "unsupported_schedule" */
  BD_BAD_PARAMS = 2,   /* "bad_params" */
  BD_MISSING_SEED = 3, /* "missing_seed" */
  BD_BAD_REQUEST = 4,  /* "bad_request" */
  BD_NO_ROOM = 5       /* "no_room" */
};

typedef struct {
  bool reinforced;
  double time;
  double magnitude;
  const char *label; /* valid until the next bd_step / bd_load / bd_unload */
} BdOutcome;

/* Build the schedule of a whole `load` line ({"schedule": {type, params}}). */
int bd_load(const char *line, size_t len);
void bd_unload(void);
int bd_reset(void);
/* Loaded spec as compact JSON, or NULL. */
const char *bd_schedule_json(void);
/* Message of the last error, or NULL. */
const char *bd_error(void);
/* Step at device time `now`; `has_event` = a response on "main" at `now`. */
int bd_step(double now, bool has_event, BdOutcome *out);
/* One conformance step from a `probe` line; writes the `outcome` line. */
int bd_probe(const char *line, size_t len, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
