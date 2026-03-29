//! Parameter schedules for inference potentials (`boltz-reference/.../schedules.py`).

use std::f64::consts::E;

/// Evaluated at `t` in `[0, 1]` (Boltz uses `steering_t = 1 - step / num_steps`).
#[derive(Debug, Clone)]
pub enum Schedule {
    Float(f64),
    ExponentialInterpolation {
        start: f64,
        end: f64,
        alpha: f64,
    },
    PiecewiseStepFunction {
        thresholds: Vec<f64>,
        values: Vec<f64>,
    },
}

impl Schedule {
    #[must_use]
    pub fn compute(&self, t: f64) -> f64 {
        match self {
            Schedule::Float(v) => *v,
            Schedule::ExponentialInterpolation { start, end, alpha } => {
                if (*alpha).abs() > f64::EPSILON {
                    *start + (end - start) * ((E.powf(alpha * t) - 1.0) / (E.powf(*alpha) - 1.0))
                } else {
                    *start + (end - start) * t
                }
            }
            Schedule::PiecewiseStepFunction { thresholds, values } => {
                assert!(!thresholds.is_empty());
                assert_eq!(values.len(), thresholds.len() + 1);
                let mut idx = 0usize;
                while idx < thresholds.len() && t > thresholds[idx] {
                    idx += 1;
                }
                values[idx]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Schedule;

    #[test]
    fn float_schedule() {
        assert!((Schedule::Float(2.5).compute(0.0) - 2.5).abs() < 1e-9);
        assert!((Schedule::Float(2.5).compute(1.0) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn exponential_interpolation_endpoints() {
        let s = Schedule::ExponentialInterpolation {
            start: 0.0,
            end: 1.0,
            alpha: -2.0,
        };
        assert!(s.compute(0.0).abs() < 1e-6);
        assert!((s.compute(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn piecewise_step() {
        let s = Schedule::PiecewiseStepFunction {
            thresholds: vec![0.25, 0.75],
            values: vec![1.0, 2.0, 3.0],
        };
        assert!((s.compute(0.0) - 1.0).abs() < 1e-9);
        assert!((s.compute(0.5) - 2.0).abs() < 1e-9);
        assert!((s.compute(1.0) - 3.0).abs() < 1e-9);
    }
}
