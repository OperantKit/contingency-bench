//! Schedule decisions for the contingency-bench Teensy firmware.
//!
//! The C++ firmware (`src/main.cpp`) keeps the serial line, the pins and
//! the clock; every reinforcement decision comes from contingency-rs
//! through the functions below (declared in `include/bench_decide.h`).
//!
//! * [`bd_load`] takes a whole `load` line, builds the schedule from its
//!   `schedule` (`{type, params}`, the conformance-fixture form) with
//!   `contingency::spec::from_spec`, and keeps it as the one loaded
//!   schedule.
//! * [`bd_step`] steps that schedule at device time `now`.
//! * [`bd_probe`] runs one conformance step from a `probe` line and writes
//!   the `outcome` line.
//!
//! The firmware is single-threaded and calls these from `loop()` only,
//! so the state is one `static` touched by one caller at a time.

#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::format;
use alloc::string::{String, ToString};
use core::ffi::{c_char, c_int};
use core::ptr::addr_of_mut;

use contingency::spec::{from_spec, SPEC_TYPES};
use contingency::{ResponseEvent, Schedule};
use serde_json::{json, Map, Value};

/// Result codes shared with `include/bench_decide.h`.
pub const BD_OK: c_int = 0;
/// The schedule `type` (or a nested one) is not a contingency-rs type.
pub const BD_UNSUPPORTED: c_int = 1;
/// The spec has the right types but parameters the constructor rejects.
pub const BD_BAD_PARAMS: c_int = 2;
/// A random schedule without `params.seed` (no_std has no entropy source).
pub const BD_MISSING_SEED: c_int = 3;
/// The line is not JSON, lacks a field, or no schedule is loaded.
pub const BD_BAD_REQUEST: c_int = 4;
/// `buf` is too small for the reply.
pub const BD_NO_ROOM: c_int = 5;

/// Schedule types that draw random numbers and so need `params.seed`.
const SEEDED_TYPES: [&str; 8] = [
    "VR",
    "RR",
    "VI",
    "RI",
    "VT",
    "RT",
    "MatchingToSample",
    "GoNoGo",
];

/// Operandum name used for device responses (the fixtures' `"main"`).
const MAIN_OPERANDUM: &str = "main";

/// Outcome of [`bd_step`] for C. `label` is NUL-terminated and stays valid
/// until the next `bd_step` / `bd_load` / `bd_unload`.
#[repr(C)]
pub struct BdOutcome {
    pub reinforced: bool,
    pub time: f64,
    pub magnitude: f64,
    pub label: *const c_char,
}

struct State {
    schedule: Option<Box<dyn Schedule>>,
    spec: Option<CString>,
    error: Option<CString>,
    label: Option<CString>,
}

static mut STATE: State = State {
    schedule: None,
    spec: None,
    error: None,
    label: None,
};

fn state() -> &'static mut State {
    // SAFETY: the firmware calls this module from `loop()` only (one
    // thread, no interrupts), so no two references are live at once.
    unsafe { &mut *addr_of_mut!(STATE) }
}

fn fail(code: c_int, msg: String) -> c_int {
    state().error = CString::new(msg.replace('\0', " ")).ok();
    code
}

fn cstring(text: &str) -> Option<CString> {
    CString::new(text.replace('\0', " ")).ok()
}

/// Read `len` bytes at `ptr` as UTF-8 JSON.
fn parse_line(ptr: *const c_char, len: usize) -> Result<Value, String> {
    if ptr.is_null() {
        return Err("null line".to_string());
    }
    // SAFETY: the caller passes a buffer of at least `len` readable bytes.
    let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    serde_json::from_slice(bytes).map_err(|e| format!("not JSON: {e}"))
}

/// Find the first nested spec whose `type` is unknown, or whose type is a
/// random one without `params.seed`. Returns the code and message.
fn screen_spec(spec: &Value) -> Option<(c_int, String)> {
    match spec {
        Value::Object(obj) => {
            if let Some(ty) = obj.get("type").and_then(Value::as_str) {
                if !SPEC_TYPES.contains(&ty) {
                    return Some((BD_UNSUPPORTED, format!("{ty} is not a contingency schedule type")));
                }
                if SEEDED_TYPES.contains(&ty) && !has_seed(obj) {
                    return Some((BD_MISSING_SEED, format!("{ty} needs params.seed on the device")));
                }
            }
            obj.values().find_map(screen_spec)
        }
        Value::Array(items) => items.iter().find_map(screen_spec),
        _ => None,
    }
}

fn has_seed(obj: &Map<String, Value>) -> bool {
    obj.get("params")
        .and_then(|p| p.get("seed"))
        .map_or(false, |s| !s.is_null())
}

/// Build and keep the schedule of a `load` line.
///
/// Returns [`BD_OK`] or an error code; the message is in [`bd_error`].
/// A failed load leaves no schedule loaded.
///
/// # Safety
///
/// `line` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn bd_load(line: *const c_char, len: usize) -> c_int {
    let st = state();
    st.schedule = None;
    st.spec = None;
    st.label = None;
    let cmd = match parse_line(line, len) {
        Ok(v) => v,
        Err(msg) => return fail(BD_BAD_REQUEST, msg),
    };
    let Some(spec) = cmd.get("schedule") else {
        return fail(BD_BAD_REQUEST, "load needs \"schedule\"".to_string());
    };
    if let Some((code, msg)) = screen_spec(spec) {
        return fail(code, msg);
    }
    match from_spec(spec) {
        Ok(schedule) => {
            st.schedule = Some(schedule);
            st.spec = cstring(&spec.to_string());
            st.error = None;
            BD_OK
        }
        Err(e) => fail(BD_BAD_PARAMS, e.to_string()),
    }
}

/// Drop the loaded schedule.
#[no_mangle]
pub extern "C" fn bd_unload() {
    let st = state();
    st.schedule = None;
    st.spec = None;
    st.label = None;
}

/// Return the loaded schedule to its post-construction state.
#[no_mangle]
pub extern "C" fn bd_reset() -> c_int {
    match state().schedule.as_mut() {
        Some(s) => {
            s.reset();
            BD_OK
        }
        None => fail(BD_BAD_REQUEST, "no schedule loaded".to_string()),
    }
}

/// The loaded schedule spec as compact JSON, or NULL when none is loaded.
#[no_mangle]
pub extern "C" fn bd_schedule_json() -> *const c_char {
    state().spec.as_ref().map_or(core::ptr::null(), |s| s.as_ptr())
}

/// The message of the last error, or NULL.
#[no_mangle]
pub extern "C" fn bd_error() -> *const c_char {
    state().error.as_ref().map_or(core::ptr::null(), |s| s.as_ptr())
}

fn step_loaded(now: f64, event: Option<ResponseEvent>) -> Result<contingency::Outcome, (c_int, String)> {
    let Some(schedule) = state().schedule.as_mut() else {
        return Err((BD_BAD_REQUEST, "no schedule loaded".to_string()));
    };
    schedule
        .step(now, event.as_ref())
        .map_err(|e| (BD_BAD_REQUEST, e.to_string()))
}

/// Step the loaded schedule at `now`; with `has_event`, a response on the
/// main operandum at `now`.
///
/// # Safety
///
/// `out` must point to a writable [`BdOutcome`].
#[no_mangle]
pub unsafe extern "C" fn bd_step(now: f64, has_event: bool, out: *mut BdOutcome) -> c_int {
    if out.is_null() {
        return fail(BD_BAD_REQUEST, "null out".to_string());
    }
    let event = has_event.then(|| ResponseEvent {
        time: now,
        operandum: MAIN_OPERANDUM.to_string(),
    });
    let empty = BdOutcome {
        reinforced: false,
        time: 0.0,
        magnitude: 0.0,
        label: core::ptr::null(),
    };
    let outcome = match step_loaded(now, event) {
        Ok(o) => o,
        Err((code, msg)) => {
            // SAFETY: checked non-null; the caller guarantees it is writable.
            unsafe { out.write(empty) };
            return fail(code, msg);
        }
    };
    let st = state();
    let written = match outcome.reinforcer {
        Some(r) => {
            st.label = cstring(&r.label);
            BdOutcome {
                reinforced: true,
                time: r.time,
                magnitude: r.magnitude,
                label: st.label.as_ref().map_or(core::ptr::null(), |s| s.as_ptr()),
            }
        }
        None => empty,
    };
    // SAFETY: as above.
    unsafe { out.write(written) };
    BD_OK
}

fn probe_reply(cmd: &Value) -> Result<String, (c_int, String)> {
    let now = cmd
        .get("now")
        .and_then(Value::as_f64)
        .ok_or((BD_BAD_REQUEST, "probe needs a number \"now\"".to_string()))?;
    let event = match cmd.get("event") {
        None | Some(Value::Null) => None,
        Some(ev) => Some(ResponseEvent {
            time: ev.get("time").and_then(Value::as_f64).unwrap_or(now),
            operandum: ev
                .get("operandum")
                .and_then(Value::as_str)
                .unwrap_or(MAIN_OPERANDUM)
                .to_string(),
        }),
    };
    let outcome = step_loaded(now, event)?;
    let reply = match outcome.reinforcer {
        Some(r) => json!({
            "kind": "outcome",
            "reinforced": true,
            "reinforcer": {"time": r.time, "magnitude": r.magnitude, "label": r.label},
        }),
        None => json!({"kind": "outcome", "reinforced": false}),
    };
    Ok(reply.to_string())
}

/// Run one conformance step from a `probe` line (`now`, `event`) and write
/// the NUL-terminated `outcome` line into `buf`.
///
/// Returns [`BD_OK`] or an error code; the message is in [`bd_error`].
///
/// # Safety
///
/// `line` must point to `len` readable bytes and `buf` to `cap` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn bd_probe(line: *const c_char, len: usize, buf: *mut c_char, cap: usize) -> c_int {
    let cmd = match parse_line(line, len) {
        Ok(v) => v,
        Err(msg) => return fail(BD_BAD_REQUEST, msg),
    };
    let reply = match probe_reply(&cmd) {
        Ok(r) => r,
        Err((code, msg)) => return fail(code, msg),
    };
    if buf.is_null() || reply.len() + 1 > cap {
        return fail(BD_NO_ROOM, format!("outcome needs {} bytes", reply.len() + 1));
    }
    // SAFETY: `buf` holds `cap` bytes and the reply plus NUL fits.
    unsafe {
        core::ptr::copy_nonoverlapping(reply.as_ptr(), buf.cast::<u8>(), reply.len());
        buf.add(reply.len()).write(0);
    }
    BD_OK
}
