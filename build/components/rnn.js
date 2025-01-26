import { in_projection, } from './rnn_weights.json';
// export const tanh = (arr: Float32Array): Float32Array => {
//     return arr.map(Math.tanh);
// };
// export const tanh2d = (arr: Float32Array[]): Float32Array[] => {
//     return arr.map(tanh);
// };
// export const sin = (arr: Float32Array): Float32Array => {
//     return arr.map(Math.sin);
// };
// export const sin2d = (arr: Float32Array[]): Float32Array[] => {
//     return arr.map(sin);
// };
// const base64ToArrayBuffer = (base64: string): ArrayBuffer => {
//     var binaryString = atob(base64);
//     var bytes = new Uint8Array(binaryString.length);
//     for (var i = 0; i < binaryString.length; i++) {
//         bytes[i] = binaryString.charCodeAt(i);
//     }
//     return bytes.buffer;
// };
// const [inProjection, inProjectionShape] = fromNpy(
//     base64ToArrayBuffer(in_projection)
// );
// const [outProjection, outProjectionShape] = fromNpy(
//     base64ToArrayBuffer(out_projection)
// );
// const [rnnInProjection, rnnInProjectionShape] = fromNpy(
//     base64ToArrayBuffer(rnn_in_projection)
// );
// const [rnnOutProjection, rnnOutProjectionShape] = fromNpy(
//     base64ToArrayBuffer(rnn_out_projection)
// );
class Rnn extends AudioWorkletProcessor {
    constructor() {
        super();
        console.log('Constructing RNN Worklet');
        console.log(in_projection);
        // console.log(inProjection, inProjectionShape);
        // console.log(outProjection, outProjectionShape);
        // console.log(rnnInProjection, rnnInProjectionShape);
        // console.log(rnnOutProjection, rnnOutProjectionShape);
        this.port.onmessage = (event) => {
            console.log('PONG', event.data);
        };
    }
    /**
     * Outputs are the first dimension, channels are the second
     */
    process(inputs, outputs, parameters) {
        for (let i = 0; i < outputs.length; i++) {
            for (let j = 0; j < outputs[i].length; j++) {
                outputs[i][j].set([Math.random() * 2 - 1]);
            }
        }
        return true;
    }
}
registerProcessor('rnn-instrument', Rnn);
//# sourceMappingURL=rnn.js.map