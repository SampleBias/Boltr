#[inline]
pub(crate) fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

#[inline]
pub(crate) fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    norm(sub(a, b))
}

#[inline]
pub(crate) fn normalize(a: [f32; 3]) -> Option<[f32; 3]> {
    let n = norm(a);
    if n > 1e-8 {
        Some(scale(a, 1.0 / n))
    } else {
        None
    }
}

/// Dihedral angle in degrees, in the range `[-180, 180]`.
pub(crate) fn dihedral_degrees(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f32 {
    let b0 = sub(p0, p1);
    let b1 = sub(p2, p1);
    let b2 = sub(p3, p2);
    let Some(b1n) = normalize(b1) else {
        return f32::NAN;
    };
    let v = sub(b0, scale(b1n, dot(b0, b1n)));
    let w = sub(b2, scale(b1n, dot(b2, b1n)));
    let x = dot(v, w);
    let y = dot(cross(b1n, v), w);
    y.atan2(x).to_degrees()
}

pub(crate) fn radius_of_gyration(coords: &[[f32; 3]]) -> Option<f32> {
    if coords.is_empty() {
        return None;
    }
    let inv_n = 1.0 / coords.len() as f32;
    let mut center = [0.0_f32; 3];
    for &c in coords {
        center = add(center, c);
    }
    center = scale(center, inv_n);
    let mut sum_sq = 0.0_f32;
    for &c in coords {
        let d = sub(c, center);
        sum_sq += dot(d, d);
    }
    Some((sum_sq * inv_n).sqrt())
}
