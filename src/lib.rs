#![allow(uncommon_codepoints)]
use {nalgebra::{Vector2, matrix, vector}, num::{sq, sqrt}, rand::{Rng, distr::Distribution, prelude::IndexedRandom}, statrs::distribution::MultivariateNormal};
#[cfg(feature="plot")] mod plot;

// Mixture of uniform and gaussian : q(h, x) = (1-h) + h*exp( -p₀(x₀-µ₀)² -p₁(x₁-µ₁)² + c )
pub struct Expert {
	mixture_weight: f64, // h
	pub(crate) mean: [f64; 2], // µ
	pub(crate) precision: [f64; 2], // p (1/σ²) axis aligned
	ln_prior_weight: f64, // c
}

pub struct Training {
	pub(crate) data : Vec<Vector2<f64>>,
	pub(crate) model : Vec<Expert>,
	pub(crate) debug : Vec<Vector2<f64>>,
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
		/*let mut model = vec![];
		for _ in 0..2 {
			let precision : f64 = Open01.sample(rng);
			model.push(Expert{mixture_weight: Open01.sample(rng), mean: vector![0., 0.], precision: vector![precision, precision], ln_prior_weight: 0.});
		}*/
		/*let model = vec![
			Expert{mixture_weight: 1., mean: [0., 0.], precision: [0.01, 0.01], ln_prior_weight: 0.},
			Expert{mixture_weight: 1., mean: [-1., -1.], precision: [4., 4.], ln_prior_weight: 10.0},
			Expert{mixture_weight: 1., mean: [1., 1.], precision: [4., 4.], ln_prior_weight: 10.0},
		];*/
		/*let model = vec![
			Expert{mixture_weight: 1., mean: [0., 0.], precision: [0.01, 0.01], ln_prior_weight: 0.},
			Expert{mixture_weight: 1., mean: [-1., -1.], precision: [2., 2.], ln_prior_weight: 10.0},
			Expert{mixture_weight: 1., mean: [1., 1.], precision: [2., 2.], ln_prior_weight: 10.0},
		];*/
		let mut model = vec![Expert{mixture_weight: 1., mean: [0., 0.], precision: [0.01, 0.01], ln_prior_weight: 0.}];
		for i in -2..=2 { model.push(Expert{mixture_weight: 1., mean: [i as f64, 0.], precision: [50., 0.2], ln_prior_weight: 10.}); }
		for j in -2..=2 { model.push(Expert{mixture_weight: 1., mean: [0., j as f64], precision: [0.2, 50.], ln_prior_weight: 10.}); }
		Self{data, model, debug: vec![]}
	}

	pub fn step(&mut self) {
		self.debug.clear();
		let Self{data, model, ..} = self;
		let ref mut rng = rand::rng();

		let mut d = *data.choose(rng).unwrap();
		for _ in 0..10000 {
			let mut sum_p = vector![0.,0.];
			let mut sum_pµ = vector![0.,0.];
			for (i, &Expert{mixture_weight: _, mean: µ, precision : p, ln_prior_weight: c}) in model.iter().enumerate() {
				// For each expert, stochastically select the gaussian or the uniform according to the posterior
				// p(hᵢ=0|x) = qᵢ(0|x)/(qᵢ(0|x)+qᵢ(1|x))
				// q(0|x) = q(x|0)*q(h=0)/q(X=x)
				// q(1|x) = q(x|1)*q(h=1)/q(X=x)
				// ... ?
				// p(hᵢ=0|x) = qᵢ(0,x)/(qᵢ(0,x)+qᵢ(1,x)) = 1/(1+qᵢ(1,x))
				//let m = MultivariateNormal::new_from_nalgebra(mean, covariance).unwrap();
				let q = f64::exp( -p[0]*sq(d[0]-µ[0]) -p[1]*sq(d[1]-µ[1]) + c );
				if rng.random_bool(1./(1.+q)) { if i>0 { continue; } } // uniform // Always keep first
				sum_p += vector![p[0], p[1]];
				sum_pµ += vector![p[0]*µ[0], p[1]*µ[1]];
			}
			if sum_p[0]==0. || sum_p[1]==0. { continue; }
			// Compute the normalized product of the selected gaussians, which is itself a gaussian, and sample from it
			d = MultivariateNormal::new_from_nalgebra(vector![sum_pµ[0]/sum_p[0], sum_pµ[1]/sum_p[1]], matrix![1./sqrt(sum_p[0]), 0.; 0., 1./sqrt(sum_p[1])]).unwrap().sample(rng);
			self.debug.push(d);
		}
	}
}
