
/// Helper function to create a linear layer without bias
///
/// This is used for projections in various modules
pub fn linear_no_bias<'a>(
    path: tch::nn::Path<'a>,
    in_dim: i64,
    out_dim: i64,
) -> tch::nn::Linear {
    tch::nn::linear(
        path,
        in_dim,
        out_dim,
        tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    )
}
