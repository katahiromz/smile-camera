// FaceDeform.ts --- 顔変形モジュール
// Author: katahiromz
// License: MIT

import { FacePoint, FaceTriangle, transferWithAffineTransform } from './Affine';
import { NormalizedLandmark } from '@mediapipe/tasks-vision';

/** 変形モードの型定義 */
export type DeformMode = 'smile' | 'bigEyes' | 'funny' | 'none';

/** 三角形の定義（ランドマークインデックスの組） */
type TriangleIndices = [number, number, number];

/**
 * 口周辺の三角形メッシュ定義
 * MediaPipe 468 landmarksから口に関連する重要な三角形のみを定義
 */
const MOUTH_TRIANGLES: TriangleIndices[] = [
  // 左口角周辺
  [61, 146, 91],     // 左口角から上方向
  [61, 185, 40],     // 左口角から上唇
  [61, 91, 181],     // 左口角下部
  
  // 右口角周辺
  [291, 375, 321],   // 右口角から上方向
  [291, 409, 270],   // 右口角から上唇
  [291, 321, 405],   // 右口角下部
  
  // 上唇
  [185, 40, 39],     // 上唇中央左
  [185, 39, 37],     // 上唇中央
  [185, 37, 267],    // 上唇中央右
  [267, 269, 409],   // 上唇右側
  
  // 下唇
  [61, 146, 91],     // 下唇左側（口角接続）
  [91, 181, 84],     // 下唇中央左
  [181, 17, 314],    // 下唇中央
  [314, 405, 321],   // 下唇中央右
  [321, 375, 291],   // 下唇右側（口角接続）
  
  // 口の内部
  [61, 40, 37],      // 口の左上
  [37, 267, 291],    // 口の右上
  [61, 37, 17],      // 口の左下
  [37, 291, 17],     // 口の中央
  [17, 291, 314],    // 口の右下
];

/**
 * 目周辺の三角形メッシュ定義
 * MediaPipe 468 landmarksから目に関連する重要な三角形のみを定義
 */
const EYE_TRIANGLES: TriangleIndices[] = [
  // 左目周辺
  [33, 160, 159],    // 左目上部（外側）
  [33, 158, 133],    // 左目下部（外側）
  [160, 159, 158],   // 左目上部（中央）
  [159, 158, 157],   // 左目中央
  [158, 157, 133],   // 左目下部（中央）
  [159, 145, 157],   // 左目上部（内側）
  [157, 145, 133],   // 左目下部（内側）
  
  // 右目周辺
  [263, 387, 386],   // 右目上部（外側）
  [263, 385, 362],   // 右目下部（外側）
  [387, 386, 385],   // 右目上部（中央）
  [386, 385, 384],   // 右目中央
  [385, 384, 362],   // 右目下部（中央）
  [386, 374, 384],   // 右目上部（内側）
  [384, 374, 362],   // 右目下部（内側）
];

/**
 * 正規化されたランドマークをピクセル座標に変換
 */
const normalizedToPixel = (
  landmark: NormalizedLandmark,
  width: number,
  height: number
): FacePoint => {
  return [landmark.x * width, landmark.y * height];
};

/**
 * 変形後のランドマーク位置を計算（smile モード）
 * 口角を上方向に移動させて笑顔にする
 */
const calculateSmileDeform = (
  landmarks: NormalizedLandmark[],
  intensity: number
): NormalizedLandmark[] => {
  const deformed = [...landmarks];
  
  // 口角のインデックス
  const LEFT_MOUTH_CORNER = 61;   // 左口角
  const RIGHT_MOUTH_CORNER = 291; // 右口角
  
  // 口角を上に移動（笑顔効果）
  const liftAmount = 0.01 * intensity; // 正規化座標での移動量
  
  deformed[LEFT_MOUTH_CORNER] = {
    ...landmarks[LEFT_MOUTH_CORNER],
    y: landmarks[LEFT_MOUTH_CORNER].y - liftAmount,
  };
  
  deformed[RIGHT_MOUTH_CORNER] = {
    ...landmarks[RIGHT_MOUTH_CORNER],
    y: landmarks[RIGHT_MOUTH_CORNER].y - liftAmount,
  };
  
  return deformed;
};

/**
 * 変形後のランドマーク位置を計算（bigEyes モード）
 * 目を縦方向に拡大する
 */
const calculateBigEyesDeform = (
  landmarks: NormalizedLandmark[],
  intensity: number
): NormalizedLandmark[] => {
  const deformed = [...landmarks];
  
  // 左目の主要ランドマーク
  const LEFT_EYE_TOP = 159;    // 左目上部
  const LEFT_EYE_BOTTOM = 145; // 左目下部
  const LEFT_EYE_CENTER = 468; // 中心は計算で求める（代替として33を使用）
  
  // 右目の主要ランドマーク
  const RIGHT_EYE_TOP = 386;    // 右目上部
  const RIGHT_EYE_BOTTOM = 374; // 右目下部
  
  const expandAmount = 0.005 * intensity; // 正規化座標での拡大量
  
  // 左目の上下を拡大
  deformed[LEFT_EYE_TOP] = {
    ...landmarks[LEFT_EYE_TOP],
    y: landmarks[LEFT_EYE_TOP].y - expandAmount,
  };
  
  deformed[LEFT_EYE_BOTTOM] = {
    ...landmarks[LEFT_EYE_BOTTOM],
    y: landmarks[LEFT_EYE_BOTTOM].y + expandAmount,
  };
  
  // 右目の上下を拡大
  deformed[RIGHT_EYE_TOP] = {
    ...landmarks[RIGHT_EYE_TOP],
    y: landmarks[RIGHT_EYE_TOP].y - expandAmount,
  };
  
  deformed[RIGHT_EYE_BOTTOM] = {
    ...landmarks[RIGHT_EYE_BOTTOM],
    y: landmarks[RIGHT_EYE_BOTTOM].y + expandAmount,
  };
  
  return deformed;
};

/**
 * 変形後のランドマーク位置を計算（funny モード）
 * 口を横に広げ、目を縦に伸ばす
 */
const calculateFunnyDeform = (
  landmarks: NormalizedLandmark[],
  intensity: number
): NormalizedLandmark[] => {
  const deformed = [...landmarks];
  
  // 口角を横に広げる
  const LEFT_MOUTH_CORNER = 61;
  const RIGHT_MOUTH_CORNER = 291;
  const widthExpand = 0.015 * intensity;
  
  deformed[LEFT_MOUTH_CORNER] = {
    ...landmarks[LEFT_MOUTH_CORNER],
    x: landmarks[LEFT_MOUTH_CORNER].x - widthExpand,
  };
  
  deformed[RIGHT_MOUTH_CORNER] = {
    ...landmarks[RIGHT_MOUTH_CORNER],
    x: landmarks[RIGHT_MOUTH_CORNER].x + widthExpand,
  };
  
  // 目を縦に伸ばす（bigEyesより強め）
  const LEFT_EYE_TOP = 159;
  const LEFT_EYE_BOTTOM = 145;
  const RIGHT_EYE_TOP = 386;
  const RIGHT_EYE_BOTTOM = 374;
  const eyeExpandAmount = 0.008 * intensity;
  
  deformed[LEFT_EYE_TOP] = {
    ...landmarks[LEFT_EYE_TOP],
    y: landmarks[LEFT_EYE_TOP].y - eyeExpandAmount,
  };
  
  deformed[LEFT_EYE_BOTTOM] = {
    ...landmarks[LEFT_EYE_BOTTOM],
    y: landmarks[LEFT_EYE_BOTTOM].y + eyeExpandAmount,
  };
  
  deformed[RIGHT_EYE_TOP] = {
    ...landmarks[RIGHT_EYE_TOP],
    y: landmarks[RIGHT_EYE_TOP].y - eyeExpandAmount,
  };
  
  deformed[RIGHT_EYE_BOTTOM] = {
    ...landmarks[RIGHT_EYE_BOTTOM],
    y: landmarks[RIGHT_EYE_BOTTOM].y + eyeExpandAmount,
  };
  
  return deformed;
};

/**
 * 変形モードに応じてランドマークを変形
 */
const getDeformedLandmarks = (
  landmarks: NormalizedLandmark[],
  mode: DeformMode,
  intensity: number
): NormalizedLandmark[] => {
  switch (mode) {
    case 'smile':
      return calculateSmileDeform(landmarks, intensity);
    case 'bigEyes':
      return calculateBigEyesDeform(landmarks, intensity);
    case 'funny':
      return calculateFunnyDeform(landmarks, intensity);
    case 'none':
    default:
      return landmarks;
  }
};

/**
 * 顔変形を適用するメイン関数
 * 口と目の周辺のみに三角形メッシュ変形を適用することで計算コストを抑える
 * 
 * @param ctx 描画対象のコンテキスト
 * @param canvas 描画対象のキャンバス（元画像を含む）
 * @param landmarks MediaPipeから取得したランドマーク（468点）
 * @param mode 変形モード ('smile' | 'bigEyes' | 'funny' | 'none')
 * @param intensity 変形の強度 (0.0-1.0, デフォルト: 0.5)
 */
export const deformFace = (
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: NormalizedLandmark[],
  mode: DeformMode = 'smile',
  intensity: number = 0.5
): void => {
  if (mode === 'none' || intensity <= 0) {
    return; // 変形なし
  }
  
  // 変形後のランドマークを計算
  const deformedLandmarks = getDeformedLandmarks(landmarks, mode, intensity);
  
  const width = canvas.width;
  const height = canvas.height;
  
  // 元画像をクローン（変形前の状態を保持）
  const originalCanvas = document.createElement('canvas');
  originalCanvas.width = width;
  originalCanvas.height = height;
  const originalCtx = originalCanvas.getContext('2d');
  if (!originalCtx) return;
  originalCtx.drawImage(canvas, 0, 0);
  
  // 対象となる三角形リスト（口と目のみ）
  const targetTriangles = [...MOUTH_TRIANGLES, ...EYE_TRIANGLES];
  
  // 各三角形に対してアフィン変換を適用
  for (const [i0, i1, i2] of targetTriangles) {
    // ランドマークが範囲外の場合はスキップ
    if (i0 >= landmarks.length || i1 >= landmarks.length || i2 >= landmarks.length) {
      continue;
    }
    
    // 元の三角形の頂点
    const srcTri: FaceTriangle = [
      normalizedToPixel(landmarks[i0], width, height),
      normalizedToPixel(landmarks[i1], width, height),
      normalizedToPixel(landmarks[i2], width, height),
    ];
    
    // 変形後の三角形の頂点
    const destTri: FaceTriangle = [
      normalizedToPixel(deformedLandmarks[i0], width, height),
      normalizedToPixel(deformedLandmarks[i1], width, height),
      normalizedToPixel(deformedLandmarks[i2], width, height),
    ];
    
    // アフィン変換を使用して三角形を変形
    transferWithAffineTransform(
      ctx,
      srcTri,
      destTri,
      (transformedCtx) => {
        // 元画像を描画
        transformedCtx.drawImage(originalCanvas, 0, 0);
      }
    );
  }
};
