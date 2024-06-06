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
vec3(0., 0., 1.),
};

const vec3 deb_colors[6] = {
vec3(1., 0.5, 0.),
vec3(0.5, 1., 0.),
vec3(1., 0., 0.5),
vec3(0.5, 0., 1.),
vec3(0., 1., 0.5),
vec3(0., 0.5, 1.)
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

vec3 get_tree_origin() {
    return vec3(ivec3(terrain_data[1], terrain_data[2], terrain_data[3]));
}

int get_eff_size(uint size) {
    return 1 << size;
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

uvec3 ascending_order(vec3 v) {
    bool a = v.x < v.y;
    bool b = v.x < v.z;
    bool c = v.y < v.z;
    return uvec3(
        uint(!(b || c)) << 0|   uint(!a && c) << 1 |    uint(a && b) << 2,
        uint(b ^^ c)  << 0|     uint(!(a ^^ c)) << 1 |  uint(a ^^ b) << 2,
        uint(b && c) << 0|      uint(a && !c) << 1 |    uint(!(a || b)) << 2
    );
}


bool get_bit(uint mask, uint n) {
    return ((mask >> n) & 1) == 1;
}

struct HeroAlgorithmPerLayerData {
    uint addr;
    vec3 position;
    vec2 s_lmax_umin;
    vec3 S_MID;
    uint CHILDMASK;
    uint LASTMASK;
    uvec3 MASKLIST;
};

HeroAlgorithmPerLayerData first_stack_entry(vec2 s_octree, vec3 S_MID) {
    return HeroAlgorithmPerLayerData(
        4, //addr
        get_tree_origin(), //position
        s_octree, //s_lmax_umin
        S_MID, //S_MID
        uint(S_MID.x < s_octree.x) << 2 | uint(S_MID.y < s_octree.x) << 1 | uint(S_MID.z < s_octree.x), //CHILDMASK
        uint(S_MID.x < s_octree.y) << 2 | uint(S_MID.y < s_octree.y) << 1 | uint(S_MID.z < s_octree.y), //LASTMASK
        ascending_order(S_MID) //MASKLIST
    );
}

vec3 cast_ray_octree() {
    const vec2 img_size = imageSize(img);
    const vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5);
    const vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;

    const vec3 ro = player.transform[3].xyz;
    //ray doesnt hit the octree at all
    const vec2 s_octree = aabb_intersection(rd, ro, get_tree_origin(), get_tree_origin() + vec3(1 << terrain_data[0]));
    if (s_octree.x > s_octree.y || s_octree.y < 0.) {
        return 0.3 * colors[min_index(1./abs(rd))];
    }
    const uint VMASK = uint(0. < rd.x) << 2 | uint(0.< rd.y) << 1 | uint(0. < rd.z);


    int stack_pointer = 0;
    HeroAlgorithmPerLayerData stack[64];
    stack[0] = first_stack_entry(s_octree, (get_tree_origin() + eff_size(terrain_data[0] - 1) - ro) / rd);
    uint failsave = 0;
    while (0 <= stack_pointer) {
        int start_stack_pointer = stack_pointer;


        if (get_bit(terrain_data[stack[start_stack_pointer].addr], stack[start_stack_pointer].CHILDMASK ^ VMASK)) {//child is leaf
            if(terrain_data[stack[start_stack_pointer].addr + 1 + (stack[start_stack_pointer].CHILDMASK ^ VMASK)] != 0) {
                return vec3(float(stack[start_stack_pointer].CHILDMASK ^ VMASK) / 8.); //Child not empty -> return color
            }
            //child empty -> continue traversal
            if (stack[start_stack_pointer].CHILDMASK == stack[start_stack_pointer].LASTMASK) {
                stack_pointer -= 1; //current node was fully traversed, return to parent node
            }
            else {
                while ((stack[start_stack_pointer].MASKLIST.x & stack[start_stack_pointer].CHILDMASK) != 0) {
                    stack[start_stack_pointer].MASKLIST = stack[start_stack_pointer].MASKLIST.yzx;
                    failsave += 1;
                    if (300 < failsave) {
                        return deb_colors[1];
                    }
                }
                stack[start_stack_pointer].CHILDMASK = stack[start_stack_pointer].CHILDMASK | stack[start_stack_pointer].MASKLIST.x;
            }
        }
        else { //child is branch
            if(stack[start_stack_pointer].MASKLIST.x == 0) {// -> Node has already been fully traversed, but couldnt be removed from stack because of childrens traversal
                stack_pointer -= 1;
            }
            else {
                uint child_identifier = (stack[start_stack_pointer].CHILDMASK ^ VMASK);
                stack_pointer += 1; //node hit is a parent -> add to stack and traverse in next iteration
                vec3 octand_child = vec3(
                    1. - float((child_identifier >> 2) & 1),
                    1. - float((child_identifier >> 1) & 1),
                    1. - float((child_identifier >> 0) & 1)
                );
                vec3 bb_min = stack[start_stack_pointer].position + octand_child * vec3(eff_size(terrain_data[0] - stack_pointer));
                vec3 bb_max = bb_min + eff_size(terrain_data[0] - stack_pointer);
                vec2 s_umin_lmax = aabb_intersection(rd, ro, bb_min, bb_max);
                vec3 S_MID = ((bb_min + bb_max) / 2 - ro) / rd;
                stack[stack_pointer] = HeroAlgorithmPerLayerData(
                    terrain_data[stack[start_stack_pointer].addr + 1 + (stack[start_stack_pointer].CHILDMASK ^ VMASK)], //addr
                    bb_min, //position
                    s_umin_lmax, //s_lmax_umin
                    S_MID, //S_MID
                    uint(S_MID.x < s_umin_lmax.x) << 2 | uint(S_MID.y < s_umin_lmax.x) << 1 | uint(S_MID.z < s_umin_lmax.x), //CHILDMASK
                    uint(S_MID.x < s_umin_lmax.y) << 2 | uint(S_MID.y < s_umin_lmax.y) << 1 | uint(S_MID.z < s_umin_lmax.y), //LASTMASK
                    ascending_order(S_MID) //MASKLIST
                );
                if (stack[start_stack_pointer].CHILDMASK == stack[start_stack_pointer].LASTMASK) { //Node has been fully traversed, but is kept on the stack as an empty member until children are also traversed
                    stack[start_stack_pointer].MASKLIST.x = 0;
                }
                else {
                    while ((stack[start_stack_pointer].MASKLIST.x & stack[start_stack_pointer].CHILDMASK) != 0) {
                        stack[start_stack_pointer].MASKLIST = stack[start_stack_pointer].MASKLIST.yzx;
                        failsave += 1;
                        if (300 < failsave) {
                            return deb_colors[0];
                        }
                    }
                    stack[start_stack_pointer].CHILDMASK = stack[start_stack_pointer].CHILDMASK | stack[start_stack_pointer].MASKLIST.x;
                }
            }
        }


        failsave += 1;
        if (300 < failsave) {
            return vec3(1., 0.4, 1.);
        }
    }




    return 0.8 * colors[min_index(1./abs(rd))];
}

void main() {

    vec4 to_write = vec4(cast_ray_octree(), 1.);
    imageStore(img, ivec2(gl_GlobalInvocationID.x, imageSize(img).y - gl_GlobalInvocationID.y), to_write);
}
