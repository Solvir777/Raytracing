#version 460
const uint CHUNK_SIZE = 64;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(push_constant) readonly uniform SetBlockData{
    ivec3 pos;
    uint value;
} block;

layout(set = 0, binding = 0, r32i) uniform iimage3D df[27];

void update_position(ivec3 pos) {
    int min_dist = int(CHUNK_SIZE);
    for (int x = 0; x < CHUNK_SIZE; x++) {
        for (int y = 0; y < CHUNK_SIZE; y++) {
            for (int z = 0; z < CHUNK_SIZE; z++) {
                bool isSolid = 0 < imageLoad(df[0], ivec3(x, y, z)).x;
                if(isSolid) {
                    int cheby_dist = max(abs(pos.x - x), max(abs(pos.y - y), abs(pos.z - z)));
                    min_dist = min(min_dist, cheby_dist);
                }
            }
        }
    }
    imageStore(df[0], pos, ivec4(min_dist));
}

void main() {
    ivec3 my_pos = ivec3(gl_GlobalInvocationID);
    int cheby_dist = max(abs(block.pos.x - my_pos.x), max(abs(block.pos.y - my_pos.y), abs(block.pos.z - my_pos.z)));
    if(block.value != 0) {
        if(my_pos == block.pos) { //this is the placed block -> change block state
            imageStore(df[0], my_pos, -ivec4(block.value));
            return;
        }
        imageAtomicMin(df[0], my_pos, cheby_dist); //-> if this is solid nothing will change, else the lesser distance will be stored
    }
    else{
        int my_value = imageLoad(df[0], my_pos).x;
        if (my_value == cheby_dist) { //this needs to find next neighbor that is near
            //update_position(my_pos);
        }
    }
}