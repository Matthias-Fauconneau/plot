#![allow(uncommon_codepoints)]
use {nalgebra::{Vector2, matrix, vector}, num::{sq, sqrt}, rand::{Rng, prelude::IndexedRandom, distr::Distribution}, statrs::distribution::MultivariateNormal};
#[cfg(feature="plot")] mod plot;

// Mixture of uniform and gaussian : q(h, x) = (1-h) + h*exp( -p₀(x₀-µ₀)² -p₁(x₁-µ₁)² + c )
pub struct Expert {
	pub(crate) mean: Vector2<f64>, // µ
	pub(crate) precision: Vector2<f64>, // p (1/σ²) axis aligned
	ln_prior_weight: f64, // c
}

pub struct Training {
	pub(crate) data : Vec<Vector2<f64>>,
	pub(crate) model : Vec<Expert>,
	pub(crate) debug : Vec<Vector2<f64>>,
}

impl Training {
	pub fn new() -> Self {
		//let mut data = vec![vector![0., 0.]];
		const N : usize = 300;
		let mut data = vec![];
		let ref mut rng = rand::rng();
		/*{
			let p : f64 = 64.;
			let source = MultivariateNormal::new_from_nalgebra(vector![0., 0.], matrix![1./p, 0.; 0., 1./p]).unwrap();
			for _ in 0..N { data.push(source.sample(rng)); }
		}*/
		for y in -2..=2 { for x in -2..=2 {
			let sigma : f64 = 1./8.;
			let source = MultivariateNormal::new_from_nalgebra(vector![x as f64, y as f64], matrix![sigma*sigma, 0.; 0., sigma*sigma]).unwrap();
			for _ in 0..N/(5*5) { data.push(source.sample(rng)); }
		} }
		/*let mut model = vec![];
		for _ in 0..2 {
			let precision : f64 = Open01.sample(rng);
			model.push(Expert{mixture_weight: Open01.sample(rng), mean: vector![0., 0.], precision: vector![precision, precision], ln_prior_weight: 0.});
		}*/
		/*let model = vec![
			Expert{mean: vector![0., 0.], precision: vector![0.01, 0.01], ln_prior_weight: 0.},
			Expert{mean: vector![-1., -1.], precision: vector![2., 2.], ln_prior_weight: 10.0},
			Expert{mean: vector![1., 1.], precision: vector![2., 2.], ln_prior_weight: 10.0},
		];*/
		/*let model = vec![
			Expert{mean: vector![0., 0.], precision: vector![0.01, 0.01], ln_prior_weight: 0.},
			Expert{mean: vector![-1., -1.], precision: vector![4., 4.], ln_prior_weight: 10.0},
			Expert{mean: vector![1., 1.], precision: vector![4., 4.], ln_prior_weight: 10.0},
		];*/
		let mut model = vec![Expert{mean: vector![0., 0.], precision: vector![0.2, 0.2], ln_prior_weight: 0.}];
		for i in -2..=2 { model.push(Expert{mean: vector![i as f64, 0.], precision: vector![200., 0.2], ln_prior_weight: 10.}); }
		for j in -2..=2 { model.push(Expert{mean: vector![0., j as f64], precision: vector![0.2, 200.], ln_prior_weight: 10.}); }
		Self{data, model, debug: vec![]}
	}

	pub fn step(&mut self) {
		self.debug.clear();
		let Self{data, model, ..} = self;
		let ref mut rng = rand::rng();

		// Gradient descent
		let mut sum_grad_c = vec![0.; model.len()];
		let mut sum_grad_p = vec![vector![0.,0.]; model.len()];
		let mut sum_grad_µ = vec![vector![0.,0.]; model.len()];
		for &d in &*data {
			for (i, &Expert{ln_prior_weight: c, precision : p, mean: µ}) in model.iter().enumerate() {
				// h ~ p(h|d): For each expert, stochastically select the gaussian or the uniform according to the posterior
				let q = f64::exp( -p[0]*sq(d[0]-µ[0]) -p[1]*sq(d[1]-µ[1]) + c );
				if rng.random_bool(1./(1.+q)) { if i>0 { continue; } } // uniform // Always keep first
				sum_grad_c[i] += 1.;
				sum_grad_p[i] += -vector![sq(d[0]-µ[0]), sq(d[1]-µ[1])];
				sum_grad_µ[i] += d-µ;
			}

			let mut d_hat = *data.choose(rng).unwrap();
			for _ in 0..100 {
				let mut sum_p = vector![0.,0.];
				let mut sum_pµ = vector![0.,0.];
				for (i, &Expert{mean: µ, precision : p, ln_prior_weight: c}) in model.iter().enumerate() {
					// h ~ p(h|d)
					let q = f64::exp( -p[0]*sq(d[0]-µ[0]) -p[1]*sq(d[1]-µ[1]) + c );
					if rng.random_bool(1./(1.+q)) { if i>0 { continue; } } // uniform // Always keep first
					sum_p += vector![p[0], p[1]];
					sum_pµ += vector![p[0]*µ[0], p[1]*µ[1]];
				}
				// Compute the normalized product of the selected gaussians, which is itself a gaussian, and sample from it
				d_hat = MultivariateNormal::new_from_nalgebra(vector![sum_pµ[0]/sum_p[0], sum_pµ[1]/sum_p[1]], matrix![1./sum_p[0], 0.; 0., 1./sum_p[1]]).unwrap().sample(rng);
			}
			self.debug.push(d_hat);

			// Same with d_hat and negative sign
			for (i, &Expert{ln_prior_weight: c, precision : p, mean: µ}) in model.iter().enumerate() {
				// h ~ p(h|d): For each expert, stochastically select the gaussian or the uniform according to the posterior
				let q = f64::exp( -p[0]*sq(d_hat[0]-µ[0]) -p[1]*sq(d_hat[1]-µ[1]) + c );
				if rng.random_bool(1./(1.+q)) { if i>0 { continue; } } // uniform // Always keep first
				sum_grad_c[i] -= 1.;
				sum_grad_p[i] -= -vector![sq(d_hat[0]-µ[0]), sq(d_hat[1]-µ[1])];
				sum_grad_µ[i] -= d_hat-µ;
			}
		}

		let learning_rate : f64 = 0.1/data.len() as f64;
		for (i, Expert{ln_prior_weight: c, precision : p, mean: µ}) in model.iter_mut().enumerate() {
			*c += learning_rate * sum_grad_c[i];
			*p += learning_rate * sum_grad_p[i] / 2.;
			p[0] = f64::max(p[0], 0.);
			p[1] = f64::max(p[1], 0.);
			*µ += learning_rate * sum_grad_µ[i];
		}
	}
}
