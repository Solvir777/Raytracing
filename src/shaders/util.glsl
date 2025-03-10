int rem_euclid_int(int dividend, int divisor) {
    int r = dividend % divisor;
    return r >= 0 ? r : r + divisor;
}

ivec3 rem_euclid_ivec3(ivec3 dividend, int divisor){
    return ivec3(rem_euclid_int(dividend.x, divisor), rem_euclid_int(dividend.y, divisor), rem_euclid_int(dividend.z, divisor));
}

vec2 chunk_AABB_test(ivec3 chunk_pos, vec3 ro, vec3 rd) {
    vec3 bmin = vec3(chunk_pos * CHUNK_SIZE);
    vec3 bmax = vec3((chunk_pos + ivec3(1)) * CHUNK_SIZE);

    vec3 inv_rd = 1. / rd;

    vec3 t1 = (bmin - ro) * inv_rd;
    vec3 t2 = (bmax - ro) * inv_rd;

    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);

    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);

    return vec2(t_enter, t_exit);
}