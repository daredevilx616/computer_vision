'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AR } from 'js-aruco';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

type TrackerBox = { x: number; y: number; width: number; height: number };
type ColorVector = { h: number; s: number; v: number };
type BackendMarker = { id: number | string; corners: [number, number][] };

const DETECT_WIDTH = 576;
const DETECT_HEIGHT = 324;

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
  const backendMarkersRef = useRef<BackendMarker[] | null>(null);
  const backendMarkerTimeRef = useRef<number>(0);

  const [markerStatus, setMarkerStatus] = useState<string>('Looking for markers...');
  const [selectedDictionary, setSelectedDictionary] = useState<string>('DICT_4X4_50');
  const backendDetectIntervalRef = useRef<number | null>(null);
  const [mode, setMode] = useState<'marker' | 'markerless' | 'sam2'>('marker');
  const markerlessRef = useRef<MarkerlessTracker>({ active: false, color: null, box: null });
  const sam2TrackerRef = useRef<MarkerlessTracker>({ active: false, color: null, box: null });
  const [markerlessStatus, setMarkerlessStatus] = useState<string>('Awaiting ROI selection');
  const [markerlessBusy, setMarkerlessBusy] = useState<boolean>(false);
  const [sam2Status, setSam2Status] = useState<string>('Upload SAM2 mask + capture frame');
  const [referenceFrame, setReferenceFrame] = useState<ImageData | null>(null);
  const maskImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    detectorRef.current = new AR.Detector();
    (async () => {
      try {
        const media = await navigator.mediaDevices.getUserMedia({ video: { width: DETECT_WIDTH, height: DETECT_HEIGHT }, audio: false });
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

  const processMarkerCanvas = useCallback(
    (canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) => {
      if (!videoRef.current) return;
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      // Always use backend markers (they're updated continuously)
      if (backendMarkersRef.current && backendMarkersRef.current.length > 0) {
        context.strokeStyle = '#00f5ff';
        context.lineWidth = 3;
        backendMarkersRef.current.forEach((marker) => {
          const corners = marker.corners;
          context.beginPath();
          context.moveTo(corners[0][0], corners[0][1]);
          corners.forEach((corner) => context.lineTo(corner[0], corner[1]));
          context.closePath();
          context.stroke();
          context.fillStyle = 'rgba(0,255,255,0.12)';
          context.fill();
          context.fillStyle = '#00f5ff';
          context.font = '12px monospace';
          context.fillText(`ID ${marker.id}`, corners[0][0] + 4, corners[0][1] + 14);
        });
      }
    },
    []
  );

  const runBackendMarkerDetect = useCallback(async () => {
    if (!markerCanvasRef.current || !videoRef.current) return;

    const canvas = markerCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const blob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob) return;

    const fd = new FormData();
    fd.append('frame', blob, 'frame.jpg');
    fd.append('dictionary', selectedDictionary);

    try {
      const res = await fetch(`${API_BASE}/api/assignment56/aruco`, { method: 'POST', body: fd });
      const payload = await res.json();
      if (res.ok) {
        backendMarkersRef.current = payload.markers || [];
        backendMarkerTimeRef.current = typeof performance !== 'undefined' ? performance.now() : Date.now();
        setMarkerStatus(payload?.message || `Detected ${payload?.count ?? 0} marker(s)`);
      }
    } catch (error: any) {
      // Silently fail - don't disrupt the UI
      console.error('Detection error:', error);
    }
  }, [selectedDictionary]);

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

  const runMarkerlessBackendStep = useCallback(async () => {
    if (!markerlessCanvasRef.current || !videoRef.current) {
      setMarkerlessStatus('Camera not ready.');
      return;
    }
    const tracker = markerlessRef.current;
    if (!tracker.box) {
      setMarkerlessStatus('Select an ROI before backend update.');
      return;
    }
    const canvas = markerlessCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const blob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
    if (!blob) {
      setMarkerlessStatus('Could not capture frame.');
      return;
    }
    const fd = new FormData();
    fd.append('frame', blob, 'frame.jpg');
    fd.append('x', tracker.box.x.toString());
    fd.append('y', tracker.box.y.toString());
    fd.append('w', tracker.box.width.toString());
    fd.append('h', tracker.box.height.toString());
    if (tracker.color) {
      fd.append('color_h', tracker.color.h.toString());
      fd.append('color_s', tracker.color.s.toString());
      fd.append('color_v', tracker.color.v.toString());
    }
    setMarkerlessBusy(true);
    setMarkerlessStatus('Backend updating tracker...');
    try {
      const res = await fetch(`${API_BASE}/api/assignment56/markerless`, { method: 'POST', body: fd });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload?.error || 'Backend update failed');
      if (payload.box) {
        tracker.box = {
          x: Number(payload.box.x),
          y: Number(payload.box.y),
          width: Number(payload.box.w),
          height: Number(payload.box.h),
        };
        tracker.active = true;
      }
      if (payload.ref_color) {
        tracker.color = {
          h: Number(payload.ref_color.h),
          s: Number(payload.ref_color.s),
          v: Number(payload.ref_color.v),
        };
      }
      setMarkerlessStatus(payload.message ?? 'Backend updated tracker.');
    } catch (error: any) {
      setMarkerlessStatus(error?.message ?? 'Backend update failed.');
    } finally {
      setMarkerlessBusy(false);
    }
  }, []);

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

  // Automatic backend detection for marker mode
  useEffect(() => {
    if (mode === 'marker') {
      // Start automatic detection loop
      const detectLoop = () => {
        runBackendMarkerDetect();
      };
      // Run immediately, then every 800ms
      detectLoop();
      backendDetectIntervalRef.current = window.setInterval(detectLoop, 800);
    } else {
      // Stop detection loop when not in marker mode
      if (backendDetectIntervalRef.current) {
        clearInterval(backendDetectIntervalRef.current);
        backendDetectIntervalRef.current = null;
      }
    }
    return () => {
      if (backendDetectIntervalRef.current) {
        clearInterval(backendDetectIntervalRef.current);
        backendDetectIntervalRef.current = null;
      }
    };
  }, [mode, runBackendMarkerDetect]);

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

    // Scale factor: canvas internal pixels vs displayed size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Convert screen coordinates to canvas pixel coordinates
    const startX = (event.clientX - rect.left) * scaleX;
    const startY = (event.clientY - rect.top) * scaleY;

    const move = (moveEvent: PointerEvent) => {
      const currentX = clamp((moveEvent.clientX - rect.left) * scaleX, 0, canvas.width);
      const currentY = clamp((moveEvent.clientY - rect.top) * scaleY, 0, canvas.height);
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
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    };
    const up = (endEvent: PointerEvent) => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      const endX = clamp((endEvent.clientX - rect.left) * scaleX, 0, canvas.width);
      const endY = clamp((endEvent.clientY - rect.top) * scaleY, 0, canvas.height);
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
        setMarkerlessStatus(`Tracking color: H=${color.h.toFixed(0)}° S=${(color.s * 100).toFixed(0)}% V=${(color.v * 100).toFixed(0)}%`);
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

    // Draw mask to canvas
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const maskData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;

    // Find bounding box of mask
    // SAM2 masks are grayscale: white (255) = object, black (0) = background
    // Check brightness (R, G, or B channel) instead of alpha
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const brightness = maskData.data[idx]; // Red channel (grayscale images have R=G=B)

        // If pixel is white (bright), it's part of the object
        if (brightness > 128) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (maxX <= minX || maxY <= minY) {
      setSam2Status('Mask invalid - no white pixels found. Make sure mask is black/white.');
      return;
    }

    const box: TrackerBox = { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
    const color = sampleColor(frame, box);

    if (!color) {
      setSam2Status('Could not derive color from mask region.');
      return;
    }

    sam2TrackerRef.current = { active: true, color, box };
    setSam2Status(`Tracking jar cap! Box: ${box.width}x${box.height}px, Color: H=${color.h.toFixed(0)}°`);
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

        <video ref={videoRef} muted playsInline className="hidden" />

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-3">
          <h2 className="text-lg font-semibold text-slate-100">Demo Video</h2>
          <iframe
            className="w-full rounded border border-slate-800"
            style={{ aspectRatio: '16 / 9', minHeight: '315px' }}
            src="https://www.youtube.com/embed/3Zwp7HGohgA"
            title="Module 5 Tracking Demo"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            frameBorder="0"
          />
          <a
            href="https://youtu.be/3Zwp7HGohgA"
            target="_blank"
            rel="noreferrer"
            className="text-sm text-emerald-200 underline hover:text-emerald-100"
          >
            Open video in a new tab
          </a>
        </section>

        
        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-3">
          <h2 className="text-lg font-semibold text-slate-100">Tracking Mode</h2>
          <div className="grid gap-3 md:grid-cols-3">
            {[
              { key: 'marker', title: 'Marker-Based', subtitle: 'ArUco/QR Detection', icon: '[]' },
              { key: 'markerless', title: 'Markerless', subtitle: 'Feature Tracking', icon: 'O' },
              { key: 'sam2', title: 'SAM2 Segmentation', subtitle: 'Pre-computed Masks', icon: '##' },
            ].map((opt) => {
              const selected = mode === opt.key;
              return (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => setMode(opt.key as 'marker' | 'markerless' | 'sam2')}
                  className={`flex flex-col items-center gap-2 rounded-xl border px-4 py-5 text-center transition ${
                    selected
                      ? 'border-emerald-400/80 bg-emerald-500/10 text-emerald-100 shadow-[0_0_0_1px_rgba(52,211,153,0.25)]'
                      : 'border-slate-700 bg-slate-900/60 text-slate-200 hover:border-slate-500 hover:bg-slate-900'
                  }`}
                >
                  <span className="text-2xl">{opt.icon}</span>
                  <span className="text-sm font-semibold">{opt.title}</span>
                  <span className="text-xs text-slate-400">{opt.subtitle}</span>
                </button>
              );
            })}
          </div>
        </section>

        {mode === 'marker' && (
          <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-slate-100">Marker Tracker (Auto-Detecting)</h2>
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <span>Dictionary:</span>
                <select
                  value={selectedDictionary}
                  onChange={(e) => setSelectedDictionary(e.target.value)}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs"
                >
                  <option value="DICT_4X4_50">4x4 (50 markers)</option>
                  <option value="DICT_4X4_100">4x4 (100 markers)</option>
                  <option value="DICT_5X5_100">5x5 (100 markers)</option>
                  <option value="DICT_5X5_250">5x5 (250 markers)</option>
                  <option value="DICT_6X6_250">6x6 (250 markers)</option>
                  <option value="DICT_ARUCO_ORIGINAL">ArUco Original</option>
                  <option value="DICT_APRILTAG_36h11">AprilTag 36h11</option>
                </select>
              </div>
            </div>
            <canvas
              ref={markerCanvasRef}
              width={DETECT_WIDTH}
              height={DETECT_HEIGHT}
              className="w-full max-w-5xl rounded border border-slate-800 bg-black"
            />
            <div className="flex items-start justify-between gap-3 text-xs text-slate-400">
              <p>{markerStatus}</p>
              <p className="text-right">Auto-detecting every 0.8s</p>
            </div>
          </section>
        )}

        {mode === 'markerless' && (
          <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-slate-100">Markerless Color Tracker</h2>
              <div className="flex items-center gap-2 text-xs text-slate-300">
                <span className="text-amber-300">👆 Drag a box over an object to track it</span>
              </div>
            </div>
            <canvas
              ref={markerlessCanvasRef}
              width={DETECT_WIDTH}
              height={DETECT_HEIGHT}
              className="w-full max-w-5xl cursor-crosshair rounded border-2 border-amber-500/30 bg-black hover:border-amber-500/50 transition-colors"
              onPointerDown={handleMarkerlessSelection}
            />
            <div className="space-y-2">
              <p className="text-xs text-slate-400">{markerlessStatus}</p>
              <div className="rounded bg-slate-800/50 p-3 text-xs text-slate-300">
                <p className="font-semibold text-amber-300 mb-1">How to use:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Hold an object still in front of the camera (use a distinctive color)</li>
                  <li>Click and drag to draw a rectangle around the object</li>
                  <li>The tracker will automatically follow it as you move the object</li>
                  <li>Works best with solid-colored objects (avoid patterns/text)</li>
                </ol>
              </div>
            </div>
          </section>
        )}

        {mode === 'sam2' && (
          <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 space-y-3">
            <h2 className="text-lg font-semibold text-slate-100">SAM2-Assisted Tracker</h2>
            <canvas
              ref={sam2CanvasRef}
              width={DETECT_WIDTH}
              height={DETECT_HEIGHT}
              className="w-full max-w-5xl rounded border border-slate-800 bg-black"
            />
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <button
                className="rounded border border-sky-500/50 px-3 py-1 text-sky-100 hover:bg-sky-500/10"
                onClick={captureReferenceFrame}
                type="button"
              >
                Capture Reference Frame
              </button>
              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={handleMaskUpload}
                className="rounded border border-slate-700 bg-slate-900/60 px-2 py-1 text-[11px]"
              />
              <span className="text-slate-400">Upload SAM2 mask + capture frame</span>
            </div>
            <p className="text-xs text-slate-400">{sam2Status}</p>
          </section>
        )}


        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 text-sm text-slate-300">
          <h2 className="text-lg font-semibold text-slate-100">Quick Tips</h2>
          <ul className="mt-2 list-disc space-y-2 pl-5 text-xs sm:text-sm">
            <li>
              <strong className="text-cyan-300">Marker tracking:</strong> Auto-detects every 0.8s. Use max brightness on phone, reduce glare.
              Try different dictionaries if not detecting. Printed markers work best!
            </li>
            <li>
              <strong className="text-amber-300">Markerless tracking:</strong> Best with solid-colored objects (bottles, cups, phones, etc).
              Drag a box around the object when still, then move it smoothly. Avoid busy backgrounds!
            </li>
            <li>
              <strong className="text-sky-300">SAM2 tracker:</strong> Capture a still frame, run SAM2 offline to get a segmentation mask,
              and upload it to initialize tracking.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
