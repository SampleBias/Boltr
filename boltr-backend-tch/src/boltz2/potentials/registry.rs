//! Build the Boltz2 inference potential list (`get_potentials(..., boltz2=True)`).

use super::schedules::Schedule;
use crate::boltz2::steering::SteeringParams;

/// One inference-time potential with schedule-backed weights (mirrors Python class + `parameters` dict).
#[derive(Debug, Clone)]
pub enum Potential {
    SymmetricChainCom {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: Schedule,
    },
    VdwOverlap {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: f64,
    },
    Connections {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: f64,
    },
    PoseBusters {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        bond_buffer: f64,
        angle_buffer: f64,
        clash_buffer: f64,
    },
    ChiralAtom {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: f64,
    },
    StereoBond {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: f64,
    },
    PlanarBond {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        buffer: f64,
    },
    Contact {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
        union_lambda: Schedule,
    },
    TemplateReference {
        guidance_interval: i64,
        guidance_weight: Schedule,
        resampling_weight: Schedule,
    },
}

#[must_use]
pub fn get_potentials_boltz2(steering: &SteeringParams) -> Vec<Potential> {
    let mut potentials = Vec::new();

    if steering.fk_steering || steering.physical_guidance_update {
        let gw = |v: f64| {
            if steering.physical_guidance_update {
                Schedule::Float(v)
            } else {
                Schedule::Float(0.0)
            }
        };

        potentials.push(Potential::SymmetricChainCom {
            guidance_interval: 4,
            guidance_weight: gw(0.5),
            resampling_weight: Schedule::Float(0.5),
            buffer: Schedule::ExponentialInterpolation {
                start: 1.0,
                end: 5.0,
                alpha: -2.0,
            },
        });

        potentials.push(Potential::VdwOverlap {
            guidance_interval: 5,
            guidance_weight: if steering.physical_guidance_update {
                Schedule::PiecewiseStepFunction {
                    thresholds: vec![0.4],
                    values: vec![0.125, 0.0],
                }
            } else {
                Schedule::Float(0.0)
            },
            resampling_weight: Schedule::PiecewiseStepFunction {
                thresholds: vec![0.6],
                values: vec![0.01, 0.0],
            },
            buffer: 0.225,
        });

        potentials.push(Potential::Connections {
            guidance_interval: 1,
            guidance_weight: gw(0.15),
            resampling_weight: Schedule::Float(1.0),
            buffer: 2.0,
        });

        potentials.push(Potential::PoseBusters {
            guidance_interval: 1,
            guidance_weight: gw(0.01),
            resampling_weight: Schedule::Float(0.1),
            bond_buffer: 0.125,
            angle_buffer: 0.125,
            clash_buffer: 0.10,
        });

        potentials.push(Potential::ChiralAtom {
            guidance_interval: 1,
            guidance_weight: gw(0.1),
            resampling_weight: Schedule::Float(1.0),
            buffer: 0.52360,
        });

        potentials.push(Potential::StereoBond {
            guidance_interval: 1,
            guidance_weight: gw(0.05),
            resampling_weight: Schedule::Float(1.0),
            buffer: 0.52360,
        });

        potentials.push(Potential::PlanarBond {
            guidance_interval: 1,
            guidance_weight: gw(0.05),
            resampling_weight: Schedule::Float(1.0),
            buffer: 0.26180,
        });
    }

    if steering.fk_steering || steering.contact_guidance_update {
        potentials.push(Potential::Contact {
            guidance_interval: 4,
            guidance_weight: if steering.contact_guidance_update {
                Schedule::PiecewiseStepFunction {
                    thresholds: vec![0.25, 0.75],
                    values: vec![0.0, 0.5, 1.0],
                }
            } else {
                Schedule::Float(0.0)
            },
            resampling_weight: Schedule::Float(1.0),
            union_lambda: Schedule::ExponentialInterpolation {
                start: 8.0,
                end: 0.0,
                alpha: -2.0,
            },
        });

        potentials.push(Potential::TemplateReference {
            guidance_interval: 2,
            guidance_weight: Schedule::Float(if steering.contact_guidance_update {
                0.1
            } else {
                0.0
            }),
            resampling_weight: Schedule::Float(1.0),
        });
    }

    potentials
}

#[cfg(test)]
mod tests {
    use super::get_potentials_boltz2;
    use crate::boltz2::steering::SteeringParams;

    #[test]
    fn get_potentials_boltz2_use_potentials_non_empty() {
        let s = SteeringParams::from_use_potentials(true);
        let pots = get_potentials_boltz2(&s);
        assert!(
            pots.len() >= 3,
            "Boltz2 branch should include symmetric chain, VDW, connections, …"
        );
    }

    #[test]
    fn get_potentials_boltz2_fast_path_empty_or_minimal() {
        let s = SteeringParams::fast_path();
        let pots = get_potentials_boltz2(&s);
        assert!(pots.is_empty(), "no FK / guidance / contact → empty list");
    }
}
