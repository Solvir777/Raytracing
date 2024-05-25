#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D img;
layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;

layout(set = 0, binding = 1, std430) buffer OctreeData {
    uint terrain_data[];
};

const vec3 colors[3] = {
    vec3(1., 0., 0.),
    vec3(0., 1., 0.),
    vec3(0., 0., 1.)
};

uint eff_size(uint a) {
    return 1 << a;
}

int min_index(vec3 i) {
    if(i.x <= i.y && i.x <= i.z) {
        return 0;
    }
    if(i.y <= i.x && i.y <= i.z) {
        return 1;
    }
    return 2;
}
vec3 cast_ray() {
    const vec2 img_size = imageSize(img);
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5);
    vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;
    vec3 ro = player.transform[3].xyz;


    vec3 inv_dir = 1. / rd;
    vec3 octand01 = step(vec3(0.), rd);
    vec3 octand11 = octand01 * 2. - 1.;
    vec3 pivot = floor(ro) + octand01;
    vec3 start_values = (pivot - ro) * inv_dir * octand11;

    vec3 dir_values = start_values;

    int m_index = 0;
    for (int i = 0; i < 75; i++) {
        m_index = min_index(dir_values * octand11);
        vec3 impact_point = ro + (((dir_values) * octand11)[m_index] + 0.001) * rd;
        if(length(floor(impact_point)) >= 30.)
        {
            return colors[m_index];
        }
        dir_values[m_index] += inv_dir[m_index];

    }
    return vec3(0.);
}

vec3 get_tree_origin() {
    return vec3(ivec3(terrain_data[1], terrain_data[2], terrain_data[3]));
}

int get_eff_size(uint size) {
    return 1 << size;
}

vec2 aabb_intersection_octree_root(vec3 rd, vec3 ro) {
    vec3 bb_min = vec3(0.);
    vec3 bb_max = vec3(1 << terrain_data[0]);

    vec3 t_min = (bb_min - ro) / rd;
    vec3 t_max = (bb_max - ro) / rd;
    vec3 t1 = min(t_min, t_max);
    vec3 t2 = max(t_min, t_max);
    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);

    return vec2(t_near, t_far);
}


vec2 aabb_intersection(vec3 rd, vec3 ro, vec3 bb_min, vec3 bb_max) {
    vec3 t_min = (bb_min - ro) / rd;
    vec3 t_max = (bb_max - ro) / rd;
    vec3 t1 = min(t_min, t_max);
    vec3 t2 = max(t_min, t_max);
    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);
    return vec2(t_near, t_far);
}

ivec3 ascending_order(vec3 v) {
    ivec3 order = ivec3(4, 2, 1);
    if (v.x > v.y) {
        v.xy = v.yx;
        order.xy = order.yx;
    }
    if (v.x > v.z) {
        v.xz = v.zx;
        order.xz = order.zx;
    }
    if (v.y > v.z) {
        v.yz = v.zy;
        order.yz = order.zy;
    }
    return order;
}


bool get_bit(uint mask, uint n) {
    return ((mask >> n) & 1) == 1;
}

vec3 cast_ray_octree() {
    const vec2 img_size = imageSize(img);
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5);
    vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;

    vec3 ro = player.transform[3].xyz - get_tree_origin();
    vec3 octand_01 = step(0., rd);
    vec3 octand_11 = octand_01 * 2 - 1;
    //ray doesnt hit the octree at all
    vec2 s_octree = aabb_intersection_octree_root(rd, ro);
    float s_l_max = s_octree.x;
    float s_u_min = s_octree.y;
    if (s_l_max > s_u_min || s_u_min < 0.) {
        return 0.3 * colors[min_index(1./abs(rd))];
    }
    vec3 to_node_center = vec3(0.) + eff_size(terrain_data[0] - 1) - ro;

    vec3 S_MID = to_node_center / rd;
    uvec3 MASKLIST = ascending_order(S_MID);
    uint CHILDMASK = uint(S_MID.x < s_l_max) << 2 | uint(S_MID.y < s_l_max) << 1 | uint(S_MID.z < s_l_max);
    uint LASTMASK = uint(S_MID.x < s_u_min) << 2 | uint(S_MID.y < s_u_min) << 1 | uint(S_MID.z < s_u_min);
    uint VMASK = uint(0. < rd.x) << 2 | uint(0.< rd.y) << 1 | uint(0. < rd.z);
    int i = 0;
    for (int i = 0; i < 4; i++) {
        if (get_bit(terrain_data[4], CHILDMASK ^ VMASK)) {
            return vec3(float(CHILDMASK ^ VMASK) / 8.);
        }
        if (CHILDMASK == LASTMASK) {
            break;
        }
        while ((MASKLIST[i] & CHILDMASK) != 0) {
            i++;

        }

        CHILDMASK = CHILDMASK | MASKLIST[i];
    }



    return 0.7 * colors[min_index(1./abs(rd))];
}

void main() {

    vec4 to_write = vec4(cast_ray_octree(), 1.);
    imageStore(img, ivec2(gl_GlobalInvocationID.x, imageSize(img).y - gl_GlobalInvocationID.y), to_write);
}
