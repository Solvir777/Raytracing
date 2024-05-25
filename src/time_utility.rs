use std::time::SystemTime;

pub struct Timer {
	config: (bool, bool, bool, f32),
	start_time: SystemTime,
	last_frame: SystemTime,
	average_delta_time: f32,
}


impl Timer {
	pub fn new(print_start_time: bool, print_delta_time: bool, print_fps: bool, smoothing_value: f32) -> Self {
		Timer{
			config: (print_start_time, print_delta_time, print_fps, smoothing_value),
			start_time: SystemTime::now(),
			last_frame: SystemTime::now(),
			average_delta_time: 0.,
		}
	}
	
	pub fn print_status(&mut self, print: bool) {
		let dt = SystemTime::now().duration_since(self.last_frame).unwrap().as_secs_f32();
		self.average_delta_time = self.average_delta_time * self.config.3 + (1. - self.config.3) * dt;
		self.last_frame = SystemTime::now();
		
		if !print {return}
		
		if self.config.0{
			println!("start_time: {}", SystemTime::now().duration_since(self.start_time).unwrap().as_secs_f32());
		}
		if self.config.1 {
			println!("delta_time: {}", self.average_delta_time);
		}
		if self.config.2 {
			println!("average fps: {}", 1. / self.average_delta_time);
		}
	}
}