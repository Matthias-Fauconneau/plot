use crate::{Training, Expert};
use nalgebra::{Vector2, matrix, vector, linalg::SymmetricEigen};
use {image::{size, xy, Image, rgb, rgba8}, line::line_no_blend as line};

impl Training {
	pub fn plot(&self, size: size) -> Image<Box<[rgba8]>> {
		let Self{data, model, ..} = self;

		let mut image = Image::fill(size, rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF});
		let size = image.size;
		let map = |p:Vector2<f64>| xy{x: (size.x-1) as f32/2. + p[0] as f32*(size.y-1) as f32/6., y: (size.y-1) as f32/2. + p[1] as f32*(size.y-1) as f32/6.};
		let plot = |mut target: Image<&mut [rgba8]>, radius:i32, rgb{r, g, b}, p:Vector2<f64>| {
			let p = xy::<i32>::from(map(p));
			//for y in -radius..=radius { if let Some(p) = (p+xy{x: 0,y}).try_unsigned() { if let Some(p) = image.get_mut(p) { *p = rgba8{r, g, b, a: 0xFF}; } } }
			//for x in -radius..=radius { if let Some(p) = (p+xy{x,y: 0}).try_unsigned() { if let Some(p) = image.get_mut(p) { *p = rgba8{r, g, b, a: 0xFF}; } } }
			for y in -radius..=radius { for x in -radius..=radius { if let Some(p) = (p+xy{x,y}).try_unsigned() { if let Some(p) = target.get_mut(p) { *p = rgba8{r, g, b, a: 0xFF}; } } } }
		};
		let line = |mut target: Image<&mut [rgba8]>, rgb{r, g, b}, p0: Vector2<f64>, p1: Vector2<f64>| line(target, map(p0), map(p1), rgba8{r, g, b, a: 0xFF});

		for Expert{mean, precision, ..} in &*model {
			let SymmetricEigen{eigenvectors, eigenvalues} = matrix![1./f64::sqrt(precision[0]), 0.; 0., 1./f64::sqrt(precision[1])].symmetric_eigen();
			//line(vector![mean[0],mean[1]]-eigenvalues[0]*eigenvectors.column(0), vector![mean[0],mean[1]]+eigenvalues[0]*eigenvectors.column(0));
			//line(vector![mean[0],mean[1]]-eigenvalues[1]*eigenvectors.column(1), vector![mean[0],mean[1]]+eigenvalues[1]*eigenvectors.column(1));
			let rect = [
				vector![mean[0],mean[1]]-eigenvalues[0]*eigenvectors.column(0)-eigenvalues[1]*eigenvectors.column(1),
				vector![mean[0],mean[1]]+eigenvalues[0]*eigenvectors.column(0)-eigenvalues[1]*eigenvectors.column(1),
				vector![mean[0],mean[1]]+eigenvalues[0]*eigenvectors.column(0)+eigenvalues[1]*eigenvectors.column(1),
				vector![mean[0],mean[1]]-eigenvalues[0]*eigenvectors.column(0)+eigenvalues[1]*eigenvectors.column(1)];
			line(image.as_mut(), rgb{r: 0, g: 0, b: 0xFF}, rect[0], rect[1]);
			line(image.as_mut(), rgb{r: 0, g: 0, b: 0xFF}, rect[1], rect[2]);
			line(image.as_mut(), rgb{r: 0, g: 0, b: 0xFF}, rect[2], rect[3]);
			line(image.as_mut(), rgb{r: 0, g: 0, b: 0xFF}, rect[3], rect[0]);
		}

		//for &p in &*data { plot(1, rgb{r: 0x80, g: 0x80, b: 0x80}, p); }
		//for (i, &p) in self.debug.iter().enumerate() { plot(i as i32, rgb{r: 0x80+i as u8, g: 0x80+i as u8, b: 0xFF}, p); }
		for &[a,b] in self.debug.array_windows() { line(image.as_mut(), rgb{r: 0x80, g: 0x80, b: 0x80}, a,b); }
		for &p in &*self.debug { plot(image.as_mut(), 1, rgb{r: 0, g: 0, b: 0}, p); }

		image
	}
}
