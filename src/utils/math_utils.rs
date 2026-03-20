use rand::Rng;

/// Randomized rounding helper modeled after `ai_tools.utils.math_utils.randomized_round`.
pub fn randomized_round(x: f64) -> i64 {
    let floor = x.floor();
    let frac = x - floor;
    let mut rng = rand::thread_rng();
    if rng.gen_bool(frac.clamp(0.0, 1.0)) {
        (floor as i64) + 1
    } else {
        floor as i64
    }
}
