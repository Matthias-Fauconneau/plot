fn main() -> image::Result {
	let mut training = training::Training::new();
	training.step();
	let image = training.plot(image::xy{x: 3840, y: 2160});
	image::save_rgb("plot.png", &image.map(|image::rgba{r,g,b,a}| image::rgb{r,g,b}))
}
