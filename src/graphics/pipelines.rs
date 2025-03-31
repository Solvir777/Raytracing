use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::ShaderStages;
use crate::graphics::push_constants::PushConstants;
use crate::graphics::shaders;

pub struct Pipelines {
    pub(crate) raytrace_pipeline: Arc<ComputePipeline>,
    pub(crate) terrain_generator_pipeline: Arc<ComputePipeline>
}

impl Pipelines {
    pub fn new(device: Arc<Device>) -> Self{
        Self{
            raytrace_pipeline: Self::create_raytrace_pipeline(device.clone()),
            terrain_generator_pipeline: Self::create_terrain_gen_pipeline(device.clone()),
        }
    }

    fn create_terrain_gen_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
        let generator_shader_module = crate::terrain::terrain_gen::generator_cs::load(device.clone()).unwrap();

        let entry_point = generator_shader_module.entry_point("main").unwrap();
        let stage_info = PipelineShaderStageCreateInfo::new(entry_point.clone());

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo{
                push_constant_ranges: vec!(PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: size_of::<nalgebra::Vector3<i32>>() as u32,
                }),
                ..PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage_info.clone()])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
            }
        ).unwrap();

        let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo{

                ..ComputePipelineCreateInfo::stage_layout(stage, layout)
            }
        ).unwrap()
    }

    fn create_raytrace_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
        let compute_shader = shaders::rt_cs::load(device.clone()).unwrap();
        let entry_point = compute_shader.entry_point("main").unwrap();
        let stage_info = PipelineShaderStageCreateInfo::new(entry_point);

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo{
                push_constant_ranges: vec![PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: size_of::<PushConstants>() as u32,
                }],
                ..PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage_info.clone()])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap()
            }
        )
            .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage_info, layout),
        )
            .unwrap()
    }
}