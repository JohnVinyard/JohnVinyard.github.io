class Interval {
  constructor(start, end) {
    this.start = start;
    this.end = end;
    this.range = end - start;
  }

  toRatio(value) {
    return (value - this.start) / this.range;
  }

  fromRatio(value) {
    return this.start + value * this.range;
  }

  translateTo(value, otherInterval) {
    const r = this.toRatio(value);
    const v = otherInterval.fromRatio(r);
    return v;
  }
}

const debug = false;

const filterCutoff = new Interval(500, 22050);
const alpha = new Interval(0, 360);
const beta = new Interval(-180, 180);
const gamma = new Interval(-90, 90);
const vertical = new Interval(0, window.innerHeight);
const unitInterval = new Interval(0, 1);

document.addEventListener("DOMContentLoaded", async (event) => {
  const start = document.getElementById("start-demo");

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

  class ConvUnit {
    initialized = false;
    gain = null;
    filt = null;

    constructor(url) {
      this.url = url;
    }

    async initialize() {
      this.initialized = true;

      const osc = context.createOscillator();
      const scriptNode = context.createScriptProcessor(512, 1, 1);
      const gainNode = context.createGain();
      const conv = context.createConvolver();

      const filter = context.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(500, context.currentTime);

      conv.buffer = await fetchAudio(this.url, context);

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
      this.filt.frequency.exponentialRampToValueAtTime(
        hz,
        context.currentTime + 0.05
      );
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

  class Controller {
    constructor(urls) {
      // this.units = urls.map(url => new ConvUnit(url));
      this.units = urls.reduce((accum, url) => {
        accum[url] = new ConvUnit(url);
        return accum;
      }, {});
    }

    updateCutoff(hz) {
      for (const key in this.units) {
        const u = this.units[key];
        u.updateCutoff(hz);
      }
    }

    async trigger(urls, amplitude) {
      urls.forEach((url) => {
        this.units[url].trigger(amplitude);
      });
    }
  }

  const activeNotes = new Set(["C"]);

  const notes = {
    C: "https://nsynth.s3.amazonaws.com/bass_electronic_018-036-100",
    E: "https://nsynth.s3.amazonaws.com/bass_electronic_018-040-127",
    G: "https://nsynth.s3.amazonaws.com/bass_electronic_018-043-100",
    B: "https://nsynth.s3.amazonaws.com/bass_electronic_018-047-100",
  };

  const unit = new Controller(Object.values(notes));

  const buttons = document.querySelectorAll(".big-button");
  buttons.forEach((button) => {
    button.addEventListener("click", (event) => {
      const id = event.target.id;
      console.log(activeNotes);
      if (activeNotes.has(id)) {
        activeNotes.delete(id);
        button.classList.remove("selected");
      } else {
        activeNotes.add(id);
        button.classList.add("selected");
      }
    });
  });

  const useMouse = () => {
    document.addEventListener(
      "mousemove",
      ({ movementX, movementY, clientX, clientY }) => {
        if (Math.abs(movementX) > 10 || Math.abs(movementY) > 10) {
          // unit.trigger(1);
          unit.trigger(
            Array.from(activeNotes).map((an) => notes[an]),
            1
          );
        }

        const u = vertical.translateTo(clientY, unitInterval);
        const hz = unitInterval.translateTo(u ** 2, filterCutoff);
        unit.updateCutoff(hz);
      }
    );
  };

  const useAcc = () => {
    if (DeviceMotionEvent) {
      if (typeof DeviceMotionEvent.requestPermission === "function") {
        DeviceMotionEvent.requestPermission();
      }

      window.addEventListener("deviceorientationabsolute", (event) => {
        const u = gamma.translateTo(event.gamma, unitInterval);
        const hz = unitInterval.translateTo(u ** 4, filterCutoff);
        unit.updateCutoff(hz);
      });

      window.addEventListener(
        "devicemotion",
        (event) => {
          const threshold = 4;

          // TODO: maybe this trigger condition should be the norm as well?
          if (
            Math.abs(event.acceleration.x) > threshold ||
            Math.abs(event.acceleration.y) > threshold ||
            Math.abs(event.acceleration.z) > threshold
          ) {
            const norm = Math.sqrt(
              event.acceleration.x ** 2 +
                event.acceleration.y ** 2 +
                event.acceleration.z ** 2
            );

            unit.trigger(
              Array.from(activeNotes).map((an) => notes[an]),
              norm * 0.2
            );
          }
        },
        true
      );
    } else {
      console.log("Device motion not supported");
      alert("device motion not supported");
    }
  };

  start.addEventListener("click", async (event) => {
    useAcc();
    useMouse();
    event.target.disabled = true;
  });
});
