use {rand::{prelude::IndexedRandom, distr::{Distribution, Open01}}, nalgebra::{matrix, vector, Vector2, Matrix2}, statrs::distribution::{Continuous, MultivariateNormal}};
#[cfg(feature="plot")] mod plot;

// Mixture of uniform and gaussian
pub struct Expert {
	weight: f64,
	pub(crate) mean: Vector2<f64>,
	pub(crate) covariance: Matrix2<f64>
}

pub struct Training {
	pub(crate) data : Vec<Vector2<f64>>,
	pub(crate) model : Vec<Expert>,
}

impl Training {
	pub fn new() -> Self {
		const N : usize = 300;
		let mut data = vec![];
		let ref mut rng = rand::rng();
		for y in -2..=2 { for x in -2..=2 {
			#[allow(non_upper_case_globals)] const sigma : f64 = 1./8.;
			let source = MultivariateNormal::new_from_nalgebra(vector![x as f64, y as f64], matrix![sigma*sigma, 0.; 0., sigma*sigma]).unwrap();
			for _ in 0..N/(5*5) { data.push(source.sample(rng)); }
		} }
		assert_eq!(data.len(), N);
		let mut model = vec![];
		for _ in 0..1 {
			let sigma : f64 = Open01.sample(rng);
			model.push(Expert{weight: 1., mean: vector![0., 0.], covariance: matrix![sigma*sigma, 0.; 0., sigma*sigma]});
		}
		Self{data, model}
	}

	pub fn step(&mut self) {
		let Self{data, model} = self;
		let ref mut rng = rand::rng();

		// Given the data, d
		let d = data.choose(rng).unwrap();
		for &Expert{weight, mean, covariance} in &*model {
			// calculate the posterior probability of selecting the Gaussian rather than the uniform in each expert
			let m = MultivariateNormal::new_from_nalgebra(mean, covariance).unwrap();
			let n = m.pdf(d);
			let p = weight*n / (weight*n + 1.);
			// only update if selected ? update weight ?
			// and compute the first term : < ∂θₘ log pₘ(d|θₘ) >Q0 (<>Q0 = expected value over data distribution)
			// <>Q0 <=> data point d ?
			// f = 1/(2π) det(Σ)⁻¹/² exp[ -1/2 (x−μ)ᵀ Σ⁻¹ (x−μ) ]
			// ∂μ ln f = Σ⁻¹ (x−μ)
			let dμ = m.precision() * (d - m.mu());
			// ∂Σ ln f = 1/2 [ Σ⁻¹ (x−μ) (x−μ)ᵀ Σ⁻¹ - Σ⁻¹ ]
			let dΣ = 1./2. * ( m.precision() * (d - m.mu()) * (d - m.mu()).transpose() * m.precision() - m.precision() );
		}
	}
}
