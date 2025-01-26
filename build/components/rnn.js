class Rnn extends AudioWorkletProcessor {
    constructor(options) {
        super();
        console.log('Constructing RNN Worklet');
        console.log(options);
        // console.log(inProjection);
        // console.log(outProjection);
        // console.log(rnnInProjection);
        // console.log(rnnOutProjection);
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