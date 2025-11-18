'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AR } from 'js-aruco';

type TrackerBox = { x: number; y: number; width: number; height: number };
type ColorVector = { h: number; s: number; v: number };

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

function rgbToHsv(r: number, g: number, b: number): ColorVector {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = ((g - b) / delta) % 6;
    else if (max === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : delta / max;
  const v = max / 255;
  return { h, s, v };
}

function colorDistance(a: ColorVector, b: ColorVector) {
  const dh = Math.min(Math.abs(a.h - b.h), 360 - Math.abs(a.h - b.h)) / 180;
  const ds = Math.abs(a.s - b.s);
  const dv = Math.abs(a.v - b.v);
  return dh * 0.6 + ds * 0.3 + dv * 0.1;
}

type MarkerlessTracker = {
  active: boolean;
  color: ColorVector | null;
  box: TrackerBox | null;
};

export default function Assignment56Page() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const markerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const markerlessCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sam2CanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<AR.Detector | null>(null);
  const animationRef = useRef<number | null>(null);

  const markerlessRef = useRef<MarkerlessTracker>({ active: false, color: null, box: null });
  const sam2TrackerRef = useRef<MarkerlessTracker>({ active: false, color: null, box: null });
  const [markerlessStatus, setMarkerlessStatus] = useState<string>('Awaiting ROI selection');
  const [sam2Status, setSam2Status] = useState<string>('Upload SAM2 mask + capture frame');
  const [referenceFrame, setReferenceFrame] = useState<ImageData | null>(null);
  const maskImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    detectorRef.current = new AR.Detector();
    (async () => {
      try {
        const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = media;
          await videoRef.current.play();
        }
      } catch (error) {
        console.error('Camera error', error);
      }
    })();
    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((track) => track.stop());
      }
    };
  }, []);

  const processMarkerCanvas = useCallback((canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) => {
    if (!videoRef.current) return;
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const frame = context.getImageData(0, 0, canvas.width, canvas.height);
    const detector = detectorRef.current;
    if (!detector) return;
    const markers = detector.detect(frame) ?? [];
    context.strokeStyle = '#0ff';
    context.lineWidth = 2;
    markers.forEach((marker) => {
      const corners = marker.corners;
      context.beginPath();
      context.moveTo(corners[0].x, corners[0].y);
      corners.forEach((corner) => context.lineTo(corner.x, corner.y));
      context.closePath();
      context.stroke();
      context.fillStyle = 'rgba(0,255,255,0.15)';
      context.fill();
    });
  }, []);

  const sampleColor = (imageData: ImageData, box: TrackerBox): ColorVector | null => {
    const { data, width } = imageData;
    const x0 = clamp(Math.floor(box.x), 0, width - 1);
    const y0 = clamp(Math.floor(box.y), 0, imageData.height - 1);
    const x1 = clamp(Math.floor(box.x + box.width), 0, width);
    const y1 = clamp(Math.floor(box.y + box.height), 0, imageData.height);
    let sumH = 0;
    let sumS = 0;
    let sumV = 0;
    let count = 0;
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        const idx = (y * width + x) * 4;
        const hsv = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
        sumH += hsv.h;
        sumS += hsv.s;
        sumV += hsv.v;
        count++;
      }
    }
    if (!count) return null;
    return { h: sumH / count, s: sumS / count, v: sumV / count };
  };

  const runAutoTracker = (imageData: ImageData, tracker: MarkerlessTracker) => {
    if (!tracker.active || !tracker.color || !tracker.box) return tracker.box;
    const { width, height, data } = imageData;
    const search = {
      x: clamp(tracker.box.x - 40, 0, width),
      y: clamp(tracker.box.y - 40, 0, height),
      width: clamp(tracker.box.width + 80, 30, width),
      height: clamp(tracker.box.height + 80, 30, height),
    };
    let sumX = 0;
    let sumY = 0;
    let hits = 0;
    for (let y = Math.floor(search.y); y < search.y + search.height; y += 3) {
      if (y >= height) break;
      for (let x = Math.floor(search.x); x < search.x + search.width; x += 3) {
        if (x >= width) break;
        const idx = (y * width + x) * 4;
        const hsv = rgbToHsv(data[idx], data[idx + 1], data[idx + 2]);
        if (colorDistance(hsv, tracker.color) < 0.25) {
          sumX += x;
          sumY += y;
          hits++;
        }
      }
    }
    if (hits < 10) return tracker.box;
    const centerX = sumX / hits;
    const centerY = sumY / hits;
    const updated: TrackerBox = {
      x: clamp(centerX - tracker.box.width / 2, 0, width - tracker.box.width),
      y: clamp(centerY - tracker.box.height / 2, 0, height - tracker.box.height),
      width: tracker.box.width,
      height: tracker.box.height,
    };
    tracker.box = updated;
    return updated;
  };

  const processTrackerCanvas = useCallback((canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, tracker: MarkerlessTracker, color: string) => {
    if (!videoRef.current) return;
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const frame = context.getImageData(0, 0, canvas.width, canvas.height);
    const box = runAutoTracker(frame, tracker);
    if (box) {
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(box.x, box.y, box.width, box.height);
    }
  }, []);

  useEffect(() => {
    const render = () => {
      const markerCanvas = markerCanvasRef.current;
      const markerlessCanvas = markerlessCanvasRef.current;
      const sam2Canvas = sam2CanvasRef.current;
      if (markerCanvas) processMarkerCanvas(markerCanvas, markerCanvas.getContext('2d')!);
      if (markerlessCanvas) processTrackerCanvas(markerlessCanvas, markerlessCanvas.getContext('2d')!, markerlessRef.current, '#fbbf24');
      if (sam2Canvas) processTrackerCanvas(sam2Canvas, sam2Canvas.getContext('2d')!, sam2TrackerRef.current, '#38bdf8');
      animationRef.current = requestAnimationFrame(render);
    };
    animationRef.current = requestAnimationFrame(render);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [processMarkerCanvas, processTrackerCanvas]);

  const handleMarkerlessSelection = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!markerlessCanvasRef.current) return;
    const canvas = markerlessCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const startX = event.clientX - rect.left;
    const startY = event.clientY - rect.top;
    const move = (moveEvent: PointerEvent) => {
      const currentX = clamp(moveEvent.clientX - rect.left, 0, canvas.width);
      const currentY = clamp(moveEvent.clientY - rect.top, 0, canvas.height);
      const box: TrackerBox = {
        x: Math.min(startX, currentX),
        y: Math.min(startY, currentY),
        width: Math.abs(currentX - startX),
        height: Math.abs(currentY - startY),
      };
      const ctx = canvas.getContext('2d');
      if (!ctx || !videoRef.current) return;
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#fbbf24';
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    };
    const up = (endEvent: PointerEvent) => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      const endX = clamp(endEvent.clientX - rect.left, 0, canvas.width);
      const endY = clamp(endEvent.clientY - rect.top, 0, canvas.height);
      const box: TrackerBox = {
        x: Math.min(startX, endX),
        y: Math.min(startY, endY),
        width: Math.abs(endX - startX),
        height: Math.abs(endY - startY),
      };
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const color = sampleColor(frame, box);
      if (color) {
        markerlessRef.current = { active: true, color, box };
        setMarkerlessStatus('Tracking target color...');
      }
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  }, []);

  const captureReferenceFrame = useCallback(() => {
    if (!sam2CanvasRef.current || !videoRef.current) return;
    const canvas = sam2CanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setReferenceFrame(data);
    setSam2Status('Reference captured. Upload SAM2 mask next.');
  }, []);

  const handleMaskUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!referenceFrame || !event.target.files?.length) {
      alert('Capture a reference frame first.');
      return;
    }
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        maskImageRef.current = img;
        applyMaskToTracker(img, referenceFrame);
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  }, [referenceFrame]);

  const applyMaskToTracker = (image: HTMLImageElement, frame: ImageData) => {
    if (!sam2CanvasRef.current) return;
    const canvas = sam2CanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const maskData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        if (maskData.data[idx + 3] > 64) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    if (maxX <= minX || maxY <= minY) {
      setSam2Status('Mask upload invalid.');
      return;
    }
    const box: TrackerBox = { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
    const color = sampleColor(frame, box);
    if (!color) {
      setSam2Status('Could not derive color from mask.');
      return;
    }
    sam2TrackerRef.current = { active: true, color, box };
    setSam2Status('Tracking using SAM2-derived ROI.');
  };

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Assignments 5-6</p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Real-Time Tracking Suite</h1>
          <p className="text-sm text-slate-300 sm:text-base">
            Demonstrate three flavors of tracking: ArUco/AprilTags, markerless color tracking, and a SAM2-initialised tracker that reuses an offline
            segmentation mask.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="grid gap-4 md:grid-cols-[minmax(0,0.5fr)_minmax(0,1fr)]">
            <div className="space-y-3">
              <video ref={videoRef} muted playsInline className="w-full rounded border border-slate-800 bg-black" />
              <p className="text-xs text-slate-400">Camera feed shared across trackers.</p>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2 text-xs">
                <p className="font-semibold text-slate-100">Marker Tracker (QR/ArUco)</p>
                <canvas
                  ref={markerCanvasRef}
                  width={320}
                  height={180}
                  className="w-full rounded border border-slate-800 bg-black"
                />
                <p>Automatic detection via js-aruco. Show at least one fiducial tag in view.</p>
              </div>
              <div className="space-y-2 text-xs">
                <p className="font-semibold text-slate-100">Markerless Tracker</p>
                <canvas
                  ref={markerlessCanvasRef}
                  width={320}
                  height={180}
                  className="w-full cursor-crosshair rounded border border-slate-800 bg-black"
                  onPointerDown={handleMarkerlessSelection}
                />
                <p className="text-slate-400">{markerlessStatus}</p>
              </div>
              <div className="space-y-2 text-xs">
                <p className="font-semibold text-slate-100">SAM2-Assisted Tracker</p>
                <canvas ref={sam2CanvasRef} width={320} height={180} className="w-full rounded border border-slate-800 bg-black" />
                <div className="space-y-1 text-[11px]">
                  <button
                    className="w-full rounded border border-sky-500/50 px-2 py-1 text-sky-100 hover:bg-sky-500/10"
                    onClick={captureReferenceFrame}
                    type="button"
                  >
                    Capture Reference Frame
                  </button>
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    onChange={handleMaskUpload}
                    className="w-full rounded border border-slate-700 bg-slate-900/60 px-2 py-1 text-[11px]"
                  />
                </div>
                <p className="text-slate-400">{sam2Status}</p>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 text-sm text-slate-300">
          <h2 className="text-lg font-semibold text-slate-100">Tips for the demonstration</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-xs sm:text-sm">
            <li>Use printed ArUco/AprilTag markers for the first tracker.</li>
            <li>For markerless tracking, drag a rectangle over your object while it is stationary, then move it for the tracker to follow.</li>
            <li>For the SAM2 tracker, capture a still frame, run SAM2 offline to get a segmentation mask, and upload it here to initialise the realtime tracker.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
