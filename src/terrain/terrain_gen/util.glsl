int rem_euclid_int(int dividend, int divisor) {
    int r = dividend % divisor;
    return r >= 0 ? r : r + divisor;
}

ivec3 rem_euclid_ivec3(ivec3 dividend, int divisor){
    return ivec3(rem_euclid_int(dividend.x, divisor), rem_euclid_int(dividend.y, divisor), rem_euclid_int(dividend.z, divisor));
}