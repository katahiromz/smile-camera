
import { FACE_MESH } from './faceMesh';
import { MESH_ANNOTATIONS } from './keypoints.ts';
import { FaceLandmarker, FilesetResolver, FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { transferWithAffineTransform } from './Affine';

// Face info type
interface FaceInfo {
  landmarks: NormalizedLandmark[];
}

let offscreenCanvas = null;

// 顔のほてりを表現
const drawFaceFeat = (ctx, x, y, radius) => {
  // 放射状のグラデーションを作る
  let rInner = 0, rOuter = radius;
  const g = ctx.createRadialGradient(x, y, rInner, x, y, rOuter);
  g.addColorStop(0, `rgba(255, 50, 50, ${Math.sin(performance.now() * 0.01) * 5 + 10}%)`);
  g.addColorStop(1, `rgba(255, 50, 50, 0%)`);

  // 半透明のグラデーション（影）を付ける
  ctx.beginPath();
  ctx.arc(x, y, rOuter, 0, 2 * Math.PI, false);
  ctx.fillStyle = g;
  ctx.fill();
};

// 顔変形を行う関数
export const deformFace = (ctx, faceInfo) => {
  // キャンバスをオフスクリーンにコピー
  let canvas = ctx.canvas;
  let width = canvas.width, height = canvas.height;
  if (!offscreenCanvas || offscreenCanvas.width < width || offscreenCanvas.height < height) {
    offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = canvas.width;
    offscreenCanvas.height = canvas.height;
  }
  const offscreenCtx = offscreenCanvas.getContext('2d', { alpha: false });
  offscreenCtx.drawImage(canvas, 0, 0);

  // 描画設定
  offscreenCtx.fillStyle = "white";       // 点の色
  offscreenCtx.font = "14px Arial";      // 番号のフォントサイズ
  offscreenCtx.lineWidth = 1;

  // 認識された各顔について
  for (const info of faceInfo) {
    const {landmarks} = info;

    // ランドマークの座標は 0.0 〜 1.0 に正規化されているため、
    // キャンバスの実際のサイズ（width, height）を掛けて座標を算出する
    const mappedLandmarks = landmarks.map((item) => {
      return { x: item.x * width, y: item.y * height };
    });

    // メッシュを描く
    if (false) {
      offscreenCtx.strokeStyle = "green";
      FACE_MESH.forEach((item) => {
        const [i0, i1, i2] = item;
        offscreenCtx.beginPath();
        offscreenCtx.moveTo(mappedLandmarks[i0].x, mappedLandmarks[i0].y);
        offscreenCtx.lineTo(mappedLandmarks[i1].x, mappedLandmarks[i1].y);
        offscreenCtx.lineTo(mappedLandmarks[i2].x, mappedLandmarks[i2].y);
        offscreenCtx.closePath();
        offscreenCtx.stroke();
      });
    }

    if (false) {
      // 頂点を描く
      offscreenCtx.fillStyle = "red";       // 点の色
      mappedLandmarks.forEach((landmark, index) => {
        const px = landmark.x, py = landmark.y;

        if (index == 10 || index == 152 || // 顔の上下
            index == 234 || index == 454 || // 顔の左右
            index == 158 || // 右目の上
            index == 385 || // 左目の上
            index == 247 || index == 467 || // 目の端
            index == 61 || index == 291 || // 口角
            index == 0 || index == 61 || index == 291 ||
            index == 17)
        {
          // 1. 頂点（小さな円）を描画
          offscreenCtx.beginPath();
          offscreenCtx.arc(px, py, 1.5, 0, 2 * Math.PI);
          offscreenCtx.fill();

          // 2. 頂点番号を描画
          offscreenCtx.fillText(index.toString(), px + 2, py - 2);
        }
      });
    }

    // 顔のスケール算出用の差分計算
    let dx0 = mappedLandmarks[10].x - mappedLandmarks[152].x;
    let dy0 = mappedLandmarks[10].y - mappedLandmarks[152].y;
    let dx1 = mappedLandmarks[454].x - mappedLandmarks[234].x;
    let dy1 = mappedLandmarks[454].y - mappedLandmarks[234].y;
    const faceWidth = Math.sqrt(dx1 * dx1 + dy1 * dy1); // 顔の横幅を基準スケールとする

    if (false) {
      // 顔の上と下
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[10].x, mappedLandmarks[10].y);
      offscreenCtx.lineTo(mappedLandmarks[152].x, mappedLandmarks[152].y);
      offscreenCtx.stroke();
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[234].x, mappedLandmarks[234].y);
      offscreenCtx.lineTo(mappedLandmarks[454].x, mappedLandmarks[454].y);
      offscreenCtx.stroke();
      // 右目の上
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[158].x, mappedLandmarks[158].y);
      offscreenCtx.lineTo(mappedLandmarks[158].x + dx0 * 0.03, mappedLandmarks[158].y + dy0 * 0.03);
      offscreenCtx.stroke();
      // 左目の上
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[385].x, mappedLandmarks[385].y);
      offscreenCtx.lineTo(mappedLandmarks[385].x + dx0 * 0.03, mappedLandmarks[385].y + dy0 * 0.03);
      offscreenCtx.stroke();
      // 右目の右端
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[247].x, mappedLandmarks[247].y);
      offscreenCtx.lineTo(mappedLandmarks[247].x - dx0 * 0.04, mappedLandmarks[247].y - dy0 * 0.04);
      offscreenCtx.stroke();
      // 左目の左端
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[467].x, mappedLandmarks[467].y);
      offscreenCtx.lineTo(mappedLandmarks[467].x - dx0 * 0.04, mappedLandmarks[467].y - dy0 * 0.04);
      offscreenCtx.stroke();
      // 口の右端
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[61].x, mappedLandmarks[61].y);
      offscreenCtx.lineTo(mappedLandmarks[61].x + dx0 * 0.03 - dx1 * 0.05, mappedLandmarks[61].y + dy0 * 0.03 - dy1 * 0.05);
      offscreenCtx.stroke();
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[43].x, mappedLandmarks[43].y);
      offscreenCtx.lineTo(mappedLandmarks[43].x + dx0 * 0.03 - dx1 * 0.05, mappedLandmarks[43].y + dy0 * 0.03 - dy1 * 0.05);
      offscreenCtx.stroke();
      // 口の左端
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[291].x, mappedLandmarks[291].y);
      offscreenCtx.lineTo(mappedLandmarks[291].x + dx0 * 0.03 + dx1 * 0.05, mappedLandmarks[291].y + dy0 * 0.03 + dy1 * 0.05);
      offscreenCtx.stroke();
      offscreenCtx.beginPath();
      offscreenCtx.moveTo(mappedLandmarks[273].x, mappedLandmarks[273].y);
      offscreenCtx.lineTo(mappedLandmarks[273].x + dx0 * 0.03 + dx1 * 0.05, mappedLandmarks[291].y + dy0 * 0.03 + dy1 * 0.05);
      offscreenCtx.stroke();
    }

    if (true) {
      // 重力（変形の中心点と移動ベクトル）の設定
      let gravity = [
        { x: mappedLandmarks[61].x, y: mappedLandmarks[61].y, ax: dx0 * 0.09 - dx1 * 0.10, ay: dy0 * 0.09 - dy1 * 0.10 },
        { x: mappedLandmarks[43].x, y: mappedLandmarks[43].y, ax: dx0 * 0.09 - dx1 * 0.10, ay: dy0 * 0.09 - dy1 * 0.10 },
        { x: mappedLandmarks[291].x, y: mappedLandmarks[291].y, ax: dx0 * 0.09 + dx1 * 0.10, ay: dy0 * 0.09 + dy1 * 0.10 },
        { x: mappedLandmarks[273].x, y: mappedLandmarks[273].y, ax: dx0 * 0.09 + dx1 * 0.10, ay: dy0 * 0.09 + dy1 * 0.10 },
        { x: mappedLandmarks[0].x, y: mappedLandmarks[0].y, ax: -dx0 * 0.015, ay: -dy0 * 0.015 },
        { x: mappedLandmarks[17].x, y: mappedLandmarks[17].y, ax: -dx0 * 0.02, ay: -dy0 * 0.02 },
      ];
      const m = 6;

      // 影響範囲の定数。顔の幅の半分程度に設定すると滑らかになります
      const sigma = faceWidth * 0.5;
      const sigma2 = sigma * sigma;

      // 変形後の座標を格納する配列を作成
      const deformedLandmarks = mappedLandmarks.map(landmark => {
        let offset = { x: 0, y: 0 };
        gravity.forEach(point => {
          const dx = landmark.x - point.x;
          const dy = landmark.y - point.y;
          const dist2 = dx * dx + dy * dy;
          if (dist2 > faceWidth * 0.3)
            return { x: landmark.x, y: landmark.y };

          // ガウス関数的な減衰を使うと、メッシュが急激に折れ曲がらず破綻しにくい
          const weight = Math.exp(-dist2 / (2 * sigma2));
          offset.x += point.ax * weight;
          offset.y += point.ay * weight;
        });
        return { x: landmark.x + offset.x, y: landmark.y + offset.y };
      });

      // メッシュの三角形群を転送
      FACE_MESH.forEach((item) => {
        const [i0, i1, i2] = item;

        // 変形前の三角形
        const triSrc: FaceTriangle = [
          [mappedLandmarks[i0].x, mappedLandmarks[i0].y],
          [mappedLandmarks[i1].x, mappedLandmarks[i1].y],
          [mappedLandmarks[i2].x, mappedLandmarks[i2].y]
        ];

        // 変形後の三角形
        const tri: FaceTriangle = [
          [deformedLandmarks[i0].x, deformedLandmarks[i0].y],
          [deformedLandmarks[i1].x, deformedLandmarks[i1].y],
          [deformedLandmarks[i2].x, deformedLandmarks[i2].y]
        ];

        // 三角形を転送
        transferWithAffineTransform(offscreenCtx, tri, ctx, triSrc);

        if (false) {
          // 変形後の三角形を描画
          offscreenCtx.beginPath();
          offscreenCtx.moveTo(tri[0][0], tri[0][1]);
          offscreenCtx.lineTo(tri[1][0], tri[1][1]);
          offscreenCtx.lineTo(tri[2][0], tri[2][1]);
          offscreenCtx.closePath();
          offscreenCtx.strokeStyle = "black";
          offscreenCtx.stroke();
        }
      });

      // 顔のほてりを表現
      drawFaceFeat(offscreenCtx, mappedLandmarks[205].x, mappedLandmarks[205].y, faceWidth * 0.2);
      drawFaceFeat(offscreenCtx, mappedLandmarks[425].x, mappedLandmarks[425].y, faceWidth * 0.2);
    }
  }

  // 元のキャンバスへ転送
  ctx.drawImage(offscreenCanvas, 0, 0);
};
