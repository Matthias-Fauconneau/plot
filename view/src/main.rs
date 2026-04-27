#![feature(slice_from_ptr_range)]
use training::Training;
use ui::{Result, run, Widget, size, int2, vulkan, shader};
use vulkan::{Context, Commands, Arc, image, PrimitiveTopology, ImageView, WriteDescriptorSet, linear};
shader!{view}

struct App {
	training : Training,
	pass: view::Pass,
}
impl App {
	fn new(context: &Context, _: &mut Commands) -> Result<Self> {
		Ok(Self{pass: view::Pass::new(context, false, PrimitiveTopology::TriangleList, false)?, training: Training::new()})
	}
}

impl Widget for App {
fn paint(&mut self, context: &Context, commands: &mut Commands, target: Arc<ImageView>, size: size, _: int2) -> Result<()> {
	let Self{pass, training} = self;
	training.step();
	let image = training.plot(size);
	let image = self::image(context, commands, image.as_ref())?;
	pass.begin_rendering(context, commands, target.clone(), None, true, &view::Uniforms::empty(), &[
		WriteDescriptorSet::image_view(0, ImageView::new_default(&image)?),
		WriteDescriptorSet::sampler(1, linear(context)),
	])?;
	unsafe{commands.draw(3, 1, 0, 0)}?;
	commands.end_rendering()?;
	Ok(())
}
fn event(&mut self, _: &Context, _: &mut Commands, _: size, _: &mut ui::EventContext, event: &ui::Event) -> Result<bool> { Ok(matches!(event, ui::Event::Idle)) }
}

fn main() -> Result { run("training", Box::new(|context, commands| Ok(Box::new(App::new(context, commands)?)))) }
