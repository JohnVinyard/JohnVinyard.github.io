class Interval {
  constructor(start, end) {
    this.start = start;
    this.end = end;
    this.range = end - start;
  }

  toRatio(value) {
    const ratio = (value - this.start) / this.range;
  }

  fromRatio(value) {
    return this.start + value * this.range;
  }

  translateTo(value, otherInterval) {
    const r = this.toRatio(value);
    const v = otherInterval.fromRatio(value);
    return v;
  }
}

const filterCutoff = new Interval(100, 1000);
const alpha = new Interval(0, 360);
const beta = new Interval(-180, 180);
const gamma = new Interval(-90, 90);

document.addEventListener("DOMContentLoaded", async (event) => {
  const start = document.getElementById("start-demo");
  const tester = document.getElementById("test");
  const recorder = document.getElementById("recorder");

  const context = new AudioContext();

  const fetchBinary = async (url) => {
    const resp = await fetch(url);
    return resp.arrayBuffer();
  };

  const audioCache = {};

  const fetchAudio = async (url, context) => {
    const cached = audioCache[url];
    if (cached !== undefined) {
      return cached;
    }

    const audioBufferPromise = fetchBinary(url).then(function (data) {
      return new Promise(function (resolve, reject) {
        context.decodeAudioData(
          data,
          (buffer) => resolve(buffer),
          (error) => reject(error)
        );
      });
    });
    audioCache[url] = audioBufferPromise;
    return audioBufferPromise;
  };

  // karplus.addEventListener("click", async () => {
  //   const osc = context.createOscillator();

  //   const scriptNode = context.createScriptProcessor(512);

  //   const mainGain = context.createGain();
  //   const delayGain = context.createGain();
  //   delayGain.gain.setValueAtTime(0.75, context.currentTime);

  //   const delay = context.createDelay(1);
  //   delay.delayTime.setValueAtTime(0.015, context.currentTime);

  //   const filter = context.createBiquadFilter();
  //   filter.type = "lowpass";
  //   filter.frequency.setValueAtTime(800, context.currentTime);

  //   osc.connect(scriptNode);
  //   scriptNode.connect(mainGain);
  //   mainGain.connect(context.destination);

  //   mainGain.connect(delay);
  //   delay.connect(delayGain);
  //   delayGain.connect(filter);
  //   filter.connect(delay);

  //   filter.connect(context.destination);

  //   mainGain.gain.exponentialRampToValueAtTime(1, context.currentTime + 0.001);
  //   mainGain.gain.exponentialRampToValueAtTime(
  //     0.000001,
  //     context.currentTime + 0.2
  //   );

  //   scriptNode.addEventListener(
  //     "audioprocess",
  //     ({ inputBuffer, outputBuffer }) => {
  //       const output = outputBuffer.getChannelData(0);

  //       for (let i = 0; i < output.length; i++) {
  //         output[i] = Math.random() * 2 - 1;
  //       }
  //     }
  //   );

  //   osc.start();

  //   setTimeout(() => {
  //     osc.disconnect(mainGain);
  //     mainGain.disconnect(scriptNode);
  //     scriptNode.disconnect(context.destination);
  //     scriptNode.disconnect(delay);
  //     delay.disconnect(context.destination);
  //     delay.disconnect(filter);
  //     filter.disconnect(scriptNode);
  //   }, 30000);
  // });

  class ConvUnit {
    initialized = false;
    gain = null;
    filt = null;

    constructor() {}

    async initialize() {
      this.initialized = true;

      const osc = context.createOscillator();
      const scriptNode = context.createScriptProcessor(512, 1, 1);
      const gainNode = context.createGain();
      const conv = context.createConvolver();
      const filter = context.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(500, context.currentTime);
      conv.buffer = await fetchAudio(
        "https://nsynth.s3.amazonaws.com/bass_electronic_018-036-100",
        context
      );

      // gainNode.gain.exponentialRampToValueAtTime(1, context.currentTime + 0.001);
      // gainNode.gain.exponentialRampToValueAtTime(
      //   0.000001,
      //   context.currentTime + 0.2
      // );

      scriptNode.addEventListener(
        "audioprocess",
        ({ inputBuffer, outputBuffer }) => {
          const output = outputBuffer.getChannelData(0);

          for (let i = 0; i < output.length; i++) {
            output[i] = Math.random() * 2 - 1;
          }
        }
      );

      osc.connect(scriptNode);
      scriptNode.connect(gainNode);
      gainNode.connect(conv);
      conv.connect(filter);
      filter.connect(context.destination);
      osc.start();

      this.gain = gainNode;
      this.filt = filter;
    }

    updateCutoff(hz) {
      this.filt.frequency.setValueAtTime(hz, context.currentTime);
    }

    async trigger(amplitude) {
      if (!this.initialized) {
        console.log("Initializing");
        await this.initialize();
      }

      this.gain.gain.exponentialRampToValueAtTime(
        amplitude,
        context.currentTime + 0.001
      );
      this.gain.gain.exponentialRampToValueAtTime(
        0.000001,
        context.currentTime + 0.2
      );
    }
  }

  const unit = new ConvUnit();

  const useMouse = () => {
    document.addEventListener("mousemove", ({ movementX, movementY }) => {
      // console.log(event);

      const x = document.getElementById("mousex");
      x.innerText = movementX;

      const y = document.getElementById("mousey");
      y.innerText = movementY;

      if (Math.abs(movementX) > 10 || Math.abs(movementY) > 10) {
        unit.trigger(1);
        recorder.innerText += ".";
      }
    });
  };

  const useAcc = () => {
    if (DeviceMotionEvent) {
      if (typeof DeviceMotionEvent.requestPermission === "function") {
        DeviceMotionEvent.requestPermission();
      }

      const motionFieldIds = ["x", "y", "z"];

      // TODO: orientation should determine the mix between three different room sounds
      const orientationFields = ["alpha", "gamma", "beta"];

      window.addEventListener("deviceorientationabsolute", (event) => {
        orientationFields.forEach((field) => {
          const el = document.getElementById(field);
          el.innerText = event[field];
        });

        const hz = gamma.translateTo(event.gamma, filterCutoff);
        unit.updateCutoff(hz);
      });

      window.addEventListener(
        "devicemotion",
        (event) => {
          motionFieldIds.forEach((field) => {
            const el = document.getElementById(field);
            el.innerText = event.acceleration[field];
          });

          if (
            Math.abs(event.acceleration.x) > 1 ||
            Math.abs(event.acceleration.y) > 1 ||
            Math.abs(event.acceleration.z) > 1
          ) {
            const norm = Math.sqrt(
              event.acceleration.x ** 2 +
                event.acceleration.y ** 2 +
                event.acceleration.z ** 2
            );
            recorder.innerText += ".";
            unit.trigger(norm * 0.5);
            // playRoomSound();
          }
        },
        true
      );
    } else {
      console.log("Device motion not supported");
      alert("device motion not supported");
    }
  };

  start.addEventListener("click", async () => {
    useAcc();
    useMouse();
  });

  tester.addEventListener("click", async () => {
    unit.trigger();
  });
});
