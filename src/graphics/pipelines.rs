use std::sync::Arc;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::ShaderStages;
use crate::graphics::push_constants::PushConstants;
use crate::graphics::shaders;
use crate::terrain::terrain_gen;

pub struct Pipelines {
    pub(crate) raytrace_pipeline: Arc<ComputePipeline>,
    pub(crate) terrain_generator_pipeline: Arc<ComputePipeline>,
    pub(crate) distance_field_pipeline: Arc<ComputePipeline>,
}

impl Pipelines {
    pub fn new(device: Arc<Device>) -> Self{
        Self{
            raytrace_pipeline: Self::create_raytrace_pipeline(device.clone()),
            terrain_generator_pipeline: Self::create_terrain_gen_pipeline(device.clone()),
            distance_field_pipeline: Self::create_distance_field_pipeline(device.clone()),
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


        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: [
                    // Binding 0: Swapchain Images (Array of Storage Images)
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            descriptor_count: 2, // Number of swapchain images (adjust if needed)
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                        },
                    ),
                    // Binding 1: block_type_image (Single Storage Image)
                    (
                        1,
                        DescriptorSetLayoutBinding {
                            descriptor_count: 1,
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                        },
                    ),
                    // Binding 2: distance_field_image (Single Storage Image)
                    (
                        2,
                        DescriptorSetLayoutBinding {
                            descriptor_count: 1,
                            stages: ShaderStages::COMPUTE,
                            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                        },
                    ),
                ].into(),
                ..Default::default()
            },
        ).unwrap();


        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo{
                push_constant_ranges: vec![PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: size_of::<PushConstants>() as u32,
                }],
                set_layouts: vec!(descriptor_set_layout),
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

    fn create_distance_field_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
        let compute_shader = terrain_gen::distance_field_cs::load(device.clone()).unwrap();
        let entry_point = compute_shader.entry_point("main").unwrap();
        
        let stage_info = PipelineShaderStageCreateInfo::new(entry_point);

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo{
                /*push_constant_ranges: vec![PushConstantRange{
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: size_of::<PushConstants>() as u32,
                }],*/
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