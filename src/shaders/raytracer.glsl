#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D render_target;

layout(set = 0, binding = 1, r32ui) readonly uniform uimage3D distance_field;


layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;

vec3 colors[] = {
        vec3(1., 0., 0.),
        vec3(0., 1., 0.),
        vec3(0., 0., 1.)
};

int min_index(vec3 i) {
    if(i.x <= i.y && i.x <= i.z) {
        return 0;
    }
    if(i.y <= i.x && i.y <= i.z) {
        return 1;
    }
    return 2;
}

vec3 aabb_intersection(vec3 rd, vec3 ro, vec3 bb_min, vec3 bb_max) {
    vec3 t_min = (bb_min - ro) / rd;
    vec3 t_max = (bb_max - ro) / rd;
    vec3 t1 = min(t_min, t_max);
    vec3 t2 = max(t_min, t_max);
    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);
    return vec3(t_near, t_far, min_index(-t1));
}

bool check_box(vec3 rd, vec3 ro) {
    vec2 minmax = aabb_intersection(rd, ro, vec3(0.), vec3(16.)).xy;
    if (minmax.x > minmax.y || minmax.x < 0) {
        return false;
    }
    return true;
}

bool in_bounds(ivec3 p) {
    const ivec2 b = ivec2(0, 15);
    return b.x <= p.x && p.x <= b.y && b.x <= p.y && p.y <= b.y && b.x <= p.z && p.z <= b.y;
}



vec3 along_ray(vec3 ro, vec3 rd, ivec3 corner) {
    return (vec3(corner) - ro) / rd;
}

vec3 cast_ray() {
    const vec2 render_img_size = imageSize(render_target);

    const vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(render_img_size.x)) - vec2(0.5, render_img_size.y / render_img_size.x * 0.5);
    const vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;
    const vec3 ro = player.transform[3].xyz;
    const ivec3 VMASK = ivec3(step((0.), rd));
    const ivec3 octand = VMASK * 2 - 1;

    vec3 aabb = aabb_intersection(rd, ro, vec3(0.), vec3(16.));


    if(aabb.x > aabb.y || aabb.y < 0.) {
        return vec3(0.1);
    }

    float overall_distance_along_ray = aabb.x;
    uint intersection_index = uint(aabb.z);


    for (int i = 0; i < 100; i++) {


        ivec3 p = ivec3(ro + overall_distance_along_ray * rd);

        p[intersection_index] = int(round(ro + overall_distance_along_ray * rd)[intersection_index]) - 1 + VMASK[intersection_index];

        if(!in_bounds(p)) {
            return vec3(0., 0., 0.5);
        }



        uint d = imageLoad(distance_field, p).x;
        if(d == 0) {
            return colors[intersection_index];
        }

        vec3 distance_to = along_ray(ro, rd, p + octand * ivec3(d) + 1 - VMASK);
        intersection_index = min_index(distance_to);
        overall_distance_along_ray = distance_to[intersection_index];

    }




    return vec3(0.2);
}

void main() {
    vec4 to_write = vec4(cast_ray(), 1.);
    imageStore(render_target, ivec2(gl_GlobalInvocationID.x, imageSize(render_target).y - gl_GlobalInvocationID.y), to_write);
}
