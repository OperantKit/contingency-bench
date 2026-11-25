//! Host tests of the C API, called as `src/main.cpp` calls it.
//!
//! The module keeps one `static` state, so every test takes `LOCK` first
//! and the tests run one at a time even under the parallel test runner.

use super::*;

use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    bd_unload();
    guard
}

fn load(line: &str) -> c_int {
    // SAFETY: `line` is a live buffer of `line.len()` bytes.
    unsafe { bd_load(line.as_ptr().cast::<c_char>(), line.len()) }
}

fn load_schedule(schedule: &str) -> c_int {
    load(&format!(r#"{{"cmd":"load","schedule":{schedule}}}"#))
}

fn step(now: f64, has_event: bool) -> (c_int, BdOutcome) {
    let mut out = BdOutcome {
        reinforced: true,
        time: -1.0,
        magnitude: -1.0,
        label: core::ptr::null(),
    };
    // SAFETY: `out` is a writable `BdOutcome`.
    let code = unsafe { bd_step(now, has_event, &mut out) };
    (code, out)
}

/// Step with a response at each time in `times` and return the indices
/// (0-based) of the responses that were reinforced.
fn reinforced_responses(times: &[f64]) -> Vec<usize> {
    times
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| {
            let (code, out) = step(t, true);
            assert_eq!(code, BD_OK, "step at {t}: {:?}", error_text());
            out.reinforced.then_some(i)
        })
        .collect()
}

fn read(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the API returns NUL-terminated strings it still owns.
    Some(unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string())
}

fn error_text() -> Option<String> {
    read(bd_error())
}

fn schedule_json() -> Option<Value> {
    read(bd_schedule_json()).map(|s| serde_json::from_str(&s).unwrap())
}

#[test]
fn vi_load_line_from_main_cpp_loads() {
    let _g = lock();
    let line = r#"{"cmd":"load","schedule":{"type":"VI","params":{"mean_interval":30,"seed":7}},"reinforcer":{"op":0,"duration":0.05},"stop_after":{"reinforcers":10,"seconds":600}}"#;
    assert_eq!(load(line), BD_OK);
    assert_eq!(error_text(), None);
    assert_eq!(
        schedule_json(),
        Some(json!({"type": "VI", "params": {"mean_interval": 30, "seed": 7}}))
    );
    // Steps without a response never reinforce on VI.
    for i in 1..=200 {
        let (code, out) = step(f64::from(i), false);
        assert_eq!(code, BD_OK);
        assert!(!out.reinforced, "no-response step at {i} s reinforced");
    }
    // The first response long after every interval has elapsed is reinforced.
    let (code, out) = step(1000.0, true);
    assert_eq!(code, BD_OK);
    assert!(out.reinforced);
    assert_eq!(out.time, 1000.0);
    assert!(read(out.label).is_some());
}

#[test]
fn fr3_reinforces_every_third_response() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"FR","params":{"n":3}}"#), BD_OK);
    assert_eq!(schedule_json(), Some(json!({"type": "FR", "params": {"n": 3}})));
    let times: Vec<f64> = (1..=9).map(f64::from).collect();
    assert_eq!(reinforced_responses(&times), vec![2, 5, 8]);
    let (code, out) = step(10.0, false);
    assert_eq!(code, BD_OK);
    assert!(!out.reinforced);
}

#[test]
fn fi10_reinforces_the_first_response_after_the_interval() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"FI","params":{"interval":10}}"#), BD_OK);
    assert_eq!(reinforced_responses(&[5.0]), Vec::<usize>::new());
    // Time passes the interval without a response: nothing is delivered.
    let (code, out) = step(11.0, false);
    assert_eq!(code, BD_OK);
    assert!(!out.reinforced);
    assert_eq!(reinforced_responses(&[12.0, 13.0]), vec![0]);
}

#[test]
fn crf_reinforces_every_response_and_nothing_else() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"CRF","params":{}}"#), BD_OK);
    for i in 1..=5 {
        let t = f64::from(i);
        let (_, none) = step(t, false);
        assert!(!none.reinforced);
        let (code, out) = step(t + 0.5, true);
        assert_eq!(code, BD_OK);
        assert!(out.reinforced);
        assert_eq!(out.time, t + 0.5);
    }
}

#[test]
fn ext_never_reinforces() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"EXT","params":{}}"#), BD_OK);
    let times: Vec<f64> = (1..=50).map(f64::from).collect();
    assert_eq!(reinforced_responses(&times), Vec::<usize>::new());
}

#[test]
fn broken_json_fails_with_a_message_and_nothing_loaded() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"FR","params":{"n":3}}"#), BD_OK);
    assert_eq!(load(r#"{"cmd":"load","schedule":{"type":"FR""#), BD_BAD_REQUEST);
    assert!(!error_text().unwrap_or_default().is_empty());
    assert_eq!(schedule_json(), None);
    let (code, out) = step(1.0, true);
    assert_eq!(code, BD_BAD_REQUEST);
    assert!(!out.reinforced);
}

#[test]
fn load_without_schedule_is_a_bad_request() {
    let _g = lock();
    assert_eq!(load(r#"{"cmd":"load"}"#), BD_BAD_REQUEST);
    assert!(error_text().unwrap().contains("schedule"));
}

#[test]
fn unknown_type_is_unsupported() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"NoSuchSchedule","params":{}}"#), BD_UNSUPPORTED);
    assert!(error_text().unwrap().contains("NoSuchSchedule"));
    assert_eq!(schedule_json(), None);
}

#[test]
fn random_schedule_without_seed_is_refused() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"VI","params":{"mean_interval":30}}"#), BD_MISSING_SEED);
    assert!(error_text().unwrap().contains("seed"));
    assert_eq!(schedule_json(), None);
}

#[test]
fn rejected_parameters_are_bad_params() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"FR","params":{"n":0}}"#), BD_BAD_PARAMS);
    assert!(!error_text().unwrap_or_default().is_empty());
    assert_eq!(schedule_json(), None);
}

#[test]
fn a_successful_load_clears_the_last_error() {
    let _g = lock();
    assert_eq!(load("not json"), BD_BAD_REQUEST);
    assert!(error_text().is_some());
    assert_eq!(load_schedule(r#"{"type":"CRF","params":{}}"#), BD_OK);
    assert_eq!(error_text(), None);
}

#[test]
fn after_unload_nothing_is_loaded() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"CRF","params":{}}"#), BD_OK);
    bd_unload();
    assert_eq!(schedule_json(), None);
    let (code, out) = step(1.0, true);
    assert_eq!(code, BD_BAD_REQUEST);
    assert!(!out.reinforced);
    assert!(error_text().unwrap().contains("no schedule loaded"));
    assert_eq!(bd_reset(), BD_BAD_REQUEST);
}

#[test]
fn reset_returns_fr_to_the_start_of_the_ratio() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"FR","params":{"n":3}}"#), BD_OK);
    assert_eq!(reinforced_responses(&[1.0, 2.0]), Vec::<usize>::new());
    assert_eq!(bd_reset(), BD_OK);
    // The schedule stays loaded, and the count starts again from 0.
    assert_eq!(schedule_json(), Some(json!({"type": "FR", "params": {"n": 3}})));
    assert_eq!(reinforced_responses(&[3.0, 4.0, 5.0]), vec![2]);
}

#[test]
fn null_out_is_a_bad_request() {
    let _g = lock();
    assert_eq!(load_schedule(r#"{"type":"CRF","params":{}}"#), BD_OK);
    // SAFETY: a null `out` is checked before any write.
    let code = unsafe { bd_step(1.0, true, core::ptr::null_mut()) };
    assert_eq!(code, BD_BAD_REQUEST);
}
