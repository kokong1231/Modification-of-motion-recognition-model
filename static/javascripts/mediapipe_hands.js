function createZeroArray(len) {
    return new Array(len).fill(0);
}

const videoElement = document.getElementsByClassName('input_video')[0];

var zeroarray = createZeroArray(84);
var point_history = new Array(60).fill(zeroarray);
var count = 0;
var mode = 0;
var model_count = 0;
var landmark_count = 0;
var detection = 0;

function onResults(results) {
    if (detection === 0) {
        if (results.multiHandedness.length > 0) {
            $(".row").html('<div class="hands_detection" style = “position:relative; z-index:1;”><div class="alert fade alert-simple alert-info text-left font__family-montserrat font__size-16 font__weight-light brk-library-rendered rendered show" role="alert" data-brk-library="component__alert"><i class="start-icon  fa fa-info-circle faa-shake animated"></i><strong class="font__weight-semibold">Scanning!</strong></div></div>')
            detection = 1;
        }
    }
    else {
        if (results.multiHandedness.length === 0) {
            $(".row").html('<div class="detection_fail" style = "position:relative; z-index:1;"><div class="alert fade alert-simple alert-warning text-left font__family-montserrat font__size-16 font__weight-light brk-library-rendered rendered show" role="alert" data-brk-library="component__alert"><i class="start-icon fa fa-exclamation-triangle faa-flash animated"></i><strong class="font__weight-semibold">Warning!</strong></div></div>')
            detection = 0;
        }
    }
    var history = createZeroArray(84);
    if (results.multiHandedness.length > 0) {
        mode = 1;
        model_count = 0;
    }
    if (mode === 1) {
        for (let i = 0; i < results.multiHandedness.length; i++) {
            if (results.multiHandedness[i].index === 0) {
                for (let j = 0; j < 21; j++) {
                    history[2 * j] = results.multiHandLandmarks[i][j].x;
                    history[2 * j + 1] = results.multiHandLandmarks[i][j].y;
                }
            }
            else {
                for (let j = 0; j < 21; j++) {
                    history[42 + 2 * j] = results.multiHandLandmarks[i][j].x
                    history[42 + 2 * j + 1] = results.multiHandLandmarks[i][j].y
                }
            }
        }
        point_history[60] = history;
        point_history.shift();
        if (results.multiHandedness.length === 0) {
            model_count += 1;
        }
        if (model_count === 20) {
            for (let k = 0; k < 60; k++) {
                let ret = point_history[k].reduce((a, b) => a + b, 0);
                if (ret > 0) {
                    landmark_count += 1;
                }
            }
            if (landmark_count >= 10) {
                modelPost();
                point_history = new Array(60).fill(zeroarray);
                mode = 0;
                model_count = 0;
                landmark_count = 0;
            }
            else {
                point_history = new Array(60).fill(zeroarray);
                mode = 0;
                model_count = 0;
                landmark_count = 0;
            }
        }
    }
}

function modelPost() {
    fetch("http://localhost:4000/api_hands", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            data: point_history,
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            // console.log("signlanglabel" + data['label']);
            console.log(data['label']);
        });
}

const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
maxNumHands: 2,
modelComplexity: 1,
minDetectionConfidence: 0.7,
minTrackingConfidence: 0.5
});

hands.onResults(onResults);


const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720
});

camera.start();