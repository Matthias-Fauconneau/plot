#![feature(slice_from_ptr_range)]
use rand_distr::{Distribution, StandardNormal};
use ui::{Result, run, Widget, size, int2, vulkan, shader};
use vulkan::{Context, Commands, Arc, Image as GPUImage, image, PrimitiveTopology, ImageView, WriteDescriptorSet, linear};
use image::{xy, Image, rgba8};
shader!{view}

struct App {
	pass: view::Pass,
	image: Arc<GPUImage>,
}
impl App {
	fn new(context: &Context, commands: &mut Commands) -> Result<Self> {
		const N : usize = 300;
		let mut points : Vec<xy<f64>> = vec![];
		let ref mut rng = rand::rng();
		for y in -2..=2 { for x in -2..=2 {
			for _ in 0..N/(5*5) {
				let sigma = 1./8.; //(2.*2.*f64::sqrt(2.*f64::ln(2.)));
				let x = x as f64 + sigma * Distribution::<f64>::sample(&StandardNormal, rng);
				let y = y as f64 + sigma * Distribution::<f64>::sample(&StandardNormal, rng);
				points.push(xy{x, y});
			}
		} }
		assert_eq!(points.len(), N);
		let mut plot = Image::fill(xy{x: 3840, y: 2160}, rgba8{r: 0, g: 0, b: 0, a: 0xFF});
		for p in points {
			let xy{x,y} = p;
			let p = xy{x: (plot.size.x as f64/2. + x*plot.size.y as f64/6.) as u32, y: (plot.size.y as f64/2. + y*plot.size.y as f64/6.) as u32};
			for y in -4..=4 { if let Some(p) = (p.signed()+xy{x: 0,y}).try_unsigned() { if let Some(p) = plot.get_mut(p) { *p = rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF}; } } }
			for x in -4..=4 { if let Some(p) = (p.signed()+xy{x,y: 0}).try_unsigned() { if let Some(p) = plot.get_mut(p) { *p = rgba8{r: 0xFF, g: 0xFF, b: 0xFF, a: 0xFF}; } } }
		}
		let image = image(context, commands, plot.as_ref())?;
		Ok(Self{pass: view::Pass::new(context, false, PrimitiveTopology::TriangleList, false)?, image})
	}
}

impl Widget for App {
fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, _: size, _: int2) -> Result<()> {
	let Self{pass, image, ..} = self;
	pass.begin_rendering(context, commands, target.clone(), None, true, &view::Uniforms::empty(), &[
		WriteDescriptorSet::image_view(0, ImageView::new_default(&image)?),
		WriteDescriptorSet::sampler(1, linear(context)),
	])?;
	unsafe{commands.draw(3, 1, 0, 0)}?;
	commands.end_rendering()?;
	Ok(())
}
}

fn main() -> Result { run("view", Box::new(|context, commands| Ok(Box::new(App::new(context, commands)?)))) }
