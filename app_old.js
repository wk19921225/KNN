let drawWidth = 280;
let drawHeight = 280;
let canvas;
let drawStatus = false;
let paint;
let color = "black";
let width = 10;
let dataObj = null;
let model = null;
$(() => {
  init()
  drawMonitor()
})

function init() {
  // 预加载数据 仅执行一次
  if (!dataObj) {
    dataObj = new getData();
    dataObj.load().then(
      () => {
        console.log(dataObj.datasetImages)
        console.log(dataObj.datasetLabels)
        model = new ML.KNN.default(dataObj.datasetImages, dataObj.datasetLabels, {k: 10})
        console.log(model)
      }
    )
  }


  for(let i = 0; i < $('.statisticalChartItem').length; i++) {
    setTimeout(() => {
      $($('.statisticalChartItem')[i]).addClass('predictActive')
    }, 100*i)
  }
  setTimeout(() => {
    for(let i = ('.statisticalChartItem').length - 1; i >= 0; i--) {
      setTimeout(() => {
        $($('.statisticalChartItem')[i]).removeClass('predictActive')
      }, 100*i)
    }
  }, 1000)

  canvas = $('#drawCanvas')[ 0 ];
  canvas.width = drawWidth;
  canvas.height = drawHeight;
  paint = canvas.getContext("2d");
  //背景色
  paint.fillStyle = "#fff";
  paint.fillRect(0, 0, drawWidth, drawHeight);
  $('.clear').on('click', () => {
    init()
  })
}

//每次手离开 向算法里推送本次绘图
function stackImgs() {
  let mycanvas = document.getElementById("drawCanvas");
  let predictCanvas = document.createElement('canvas');
  predictCanvas.width = 28
  predictCanvas.height = 28
  let ctx = predictCanvas.getContext('2d');
  ctx.drawImage(mycanvas, 0, 0, 28, 28)
  ctx.scale(0.1, 0.1)
  let predictImageData = ctx.getImageData(0, 0, 28, 28)
  // console.log(predictImageData)
  // 将预测数据色值 R 取出，色值除以 255，得到新的 ArrayBuffer
  const predictImageBuffer =
    new ArrayBuffer(predictImageData.width * predictImageData.height);
  for(let i = 0; i < predictImageData.data.length / 4; i++) {
    predictImageBuffer[i] = predictImageData.data[4*i] / 255 === 1 ? 0 : 1;
  }
  // let result = model.predict([predictImageBuffer])
  // console.log(result)
}

let startX, startY;

function drawMonitor() {
  //给画笔添加上个事件一个点击开始 ， 点击后移动 ，点击事件结束
  $(canvas).on("mousedown mousemove mouseup", function ( event ) {
    let endX;
    let endY;
    switch ( event.type ) {
      case "mousedown":
        // 开启绘画状态
        // console.log(event)
        drawStatus = true;
        //记录触屏的第一个点
        startX = event.offsetX;
        startY = event.offsetY;
        break;
      case "mousemove":
        if ( drawStatus ) {
          // console.log(event)
          event.preventDefault();
          endX = event.offsetX;
          endY = event.offsetY;

          //画下线段
          paint.beginPath();
          paint.moveTo(startX, startY);
          paint.lineTo(endX, endY);
          paint.closePath();
          //动态的设置颜色
          paint.strokeStyle = color;
          paint.lineWidth = width;
          paint.stroke();
          startX = endX;
          startY = endY;
        }
        break;
      //手离开触屏是橡皮檫隐藏
      case "mouseup":
        // console.log(event)
        // 关闭绘画状态
        drawStatus = false;
        stackImgs();
        break;
    }
  });
}

// 获取数据类
function getData( ) {
  const IMAGE_SIZE = 28 * 28;
  const NUM_CLASSES = 10;
  const NUM_DATASET_ELEMENTS = 65000;
  const NUM_TRAIN_ELEMENTS = 55000;
  const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
  const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
  const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
  return {
    datasetImages: null,
    datasetLabels: null,
    trainImages: null,
    testImages: null,
    trainLabels: null,
    testLabels: null,
    load: async function () {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const imgRequest = new Promise((resolve, reject) => {
        img.crossOrigin = '';
        img.src = MNIST_IMAGES_SPRITE_PATH;
        img.onload = () => {
          img.width = img.naturalWidth;
          img.height = img.naturalHeight;
          // console.log(img.naturalWidth, img.naturalHeight)
          const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

          const chunkSize = 5000;
          canvas.width = img.width;
          canvas.height = chunkSize;

          for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
            const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
            ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (let j = 0; j < imageData.data.length / 4; j++) {
              // R 位置色值 / 255
              datasetBytesView[j] = imageData.data[j * 4] / 255;
            }
          }
          let datasetImages = new Float32Array(datasetBytesBuffer);
          this.datasetImages = getLayoutData(datasetImages, 784).slice(0, 10)
          resolve();
        };
      });
      const labelsRequest = fetch(MNIST_LABELS_PATH);
      const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);
      // console.log(labelsResponse)
      let datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
      this.datasetLabels = getLayoutData(datasetLabels, 10).slice(0, 10);
      // this.trainImages =
      //   this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      // this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
      // this.trainLabels =
      //   this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
      // this.testLabels =
      //   this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }
  }
}

function getLayoutData(typedArray, num) {
  let returnArr = [];
  let tempArr = [];
  for (let index = 0; index < typedArray.length; index++) {
    if (index % num === 0) {
      returnArr.push(tempArr)
    } else {
      tempArr.push(typedArray[index])
    }
  }
  return returnArr;
}
