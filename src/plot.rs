use crate::{Training, Expert};
use nalgebra::{Vector2, matrix, linalg::SymmetricEigen};
use {image::{size, xy, Image, rgba8}, line::line_no_blend as line};

impl Training {
	pub fn plot(&self, size: size) -> Image<Box<[rgba8]>> {
		let Self{data, model} = self;

		let mut image = Image::fill(size, rgba8{r: 0, g: 0, b: 0, a: 0xFF});
		let size = image.size;
		let map = |p:Vector2<f64>| xy{x: (size.x as f32/2. + p[0] as f32*size.y as f32/6.), y: (size.y as f32/2. + p[1] as f32*size.y as f32/6.)};

		for &p in &*data {
			let p = xy::<i32>::from(map(p));
			for y in -4..=4 { if let Some(p) = (p+xy{x: 0,y}).try_unsigned() { if let Some(p) = image.get_mut(p) { *p = rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF}; } } }
			for x in -4..=4 { if let Some(p) = (p+xy{x,y: 0}).try_unsigned() { if let Some(p) = image.get_mut(p) { *p = rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF}; } } }
		}

		for Expert{mean, precision, ..} in &*model {
			let SymmetricEigen{eigenvectors, eigenvalues} = matrix![1./f64::sqrt(precision[0]), 0.; 0., 1./f64::sqrt(precision[1])].symmetric_eigen();
			let mut line = |a:Vector2<f64>, b:Vector2<f64>| line(image.as_mut(), map(a), map(b), rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF});
			line(mean-eigenvalues[0]*eigenvectors.column(0), mean+eigenvalues[0]*eigenvectors.column(0));
			line(mean-eigenvalues[1]*eigenvectors.column(1), mean+eigenvalues[1]*eigenvectors.column(1));
		}

		image
	}
}
