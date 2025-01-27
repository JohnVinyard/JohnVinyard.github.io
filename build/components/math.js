export const twoDimArray = (data, shape) => {
    const [x, y] = shape;
    const output = [];
    for (let i = 0; i < data.length; i += y) {
        output.push(data.slice(i, i + y));
    }
    return output;
};
export const vectorVectorDot = (a, b) => {
    return a.reduce((accum, current, index) => {
        return accum + current * b[index];
    }, 0);
};
export const elementwiseSum = (a, b) => {
    return a.map((value, index) => value + b[index]);
};
export const sum = (a) => {
    return a.reduce((accum, current) => {
        return accum + current;
    }, 0);
};
export const l1Norm = (a) => {
    return sum(a.map(Math.abs));
};
/**
 * e.g., if vetor is length 64, and matrix is (128, 64), we'll end up
 * with a new vector of length 128
 */
export const dotProduct = (vector, matrix) => {
    return new Float32Array(matrix.map((v) => vectorVectorDot(v, vector)));
};
//# sourceMappingURL=math.js.map