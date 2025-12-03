'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  DrawingUtils,
} from '@mediapipe/tasks-vision';

type Vertex = { x: number; y: number };

type StereoResult = {
  disparity: string;
  segments: Array<{ start: { x: number; y: number; z_mm: number }; end: { x: number; y: number; z_mm: number }; length_mm: number }>;
  mean_length_mm: number;
};

type CsvRow = { timestamp: number; type: string; index: number; x: number; y: number; z: number; visibility?: number };

export default function Assignment7Page() {
  const [leftFile, setLeftFile] = useState<File | null>(null);
  const [rightFile, setRightFile] = useState<File | null>(null);
  const [vertices, setVertices] = useState<Vertex[]>([]);
  const [scale, setScale] = useState<number>(1);
  const [stereoStatus, setStereoStatus] = useState<string>('Awaiting inputs');
  const [stereoResult, setStereoResult] = useState<StereoResult | null>(null);
  const drawingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewImageRef = useRef<HTMLImageElement | null>(null);

  const loadPreview = useCallback((file: File | null) => {
    if (!file || !drawingCanvasRef.current) return;
    const canvas = drawingCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      const maxWidth = 640;
      const ratio = img.width > maxWidth ? maxWidth / img.width : 1;
      const width = img.width * ratio;
      const height = img.height * ratio;
      canvas.width = width;
      canvas.height = height;
      setScale(1 / ratio);
      ctx.drawImage(img, 0, 0, width, height);
      previewImageRef.current = img;
      setVertices([]);
    };
    img.src = URL.createObjectURL(file);
  }, []);

  const handleLeftChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null;
      setLeftFile(file);
      loadPreview(file);
    },
    [loadPreview],
  );

  const handleRightChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setRightFile(file);
  }, []);

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!drawingCanvasRef.current) return;
      const rect = drawingCanvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      setVertices((prev) => [...prev, { x, y }]);
    },
    [],
  );

  useEffect(() => {
    if (!drawingCanvasRef.current) return;
    const canvas = drawingCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (previewImageRef.current) {
      ctx.drawImage(previewImageRef.current, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = 'rgba(15,23,42,0.8)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    vertices.forEach((vertex, index) => {
      ctx.fillStyle = '#22d3ee';
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(String(index + 1), vertex.x + 6, vertex.y - 6);
    });
    if (vertices.length >= 2) {
      ctx.strokeStyle = '#22d3ee';
      ctx.beginPath();
      ctx.moveTo(vertices[0].x, vertices[0].y);
      vertices.slice(1).forEach((vertex) => ctx.lineTo(vertex.x, vertex.y));
      ctx.closePath();
      ctx.stroke();
    }
  }, [vertices]);

  const runStereoMeasurement = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!leftFile || !rightFile || vertices.length < 2) {
      alert('Upload stereo pair and mark the polygon.');
      return;
    }
    const form = event.currentTarget;
    const baseline = Number((form.elements.namedItem('baselineMm') as HTMLInputElement).value);
    const focal = Number((form.elements.namedItem('focalMm') as HTMLInputElement).value);
    const sensorWidth = Number((form.elements.namedItem('sensorWidthMm') as HTMLInputElement).value);
    const dataVertices = vertices.map((vertex) => ({ x: vertex.x * scale, y: vertex.y * scale }));
    const fd = new FormData();
    fd.append('left', leftFile);
    fd.append('right', rightFile);
    fd.append('baselineMm', String(baseline));
    fd.append('focalMm', String(focal));
    fd.append('sensorWidthMm', String(sensorWidth));
    fd.append('polygon', JSON.stringify(dataVertices));
    setStereoStatus('Running stereo reconstruction...');
    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';
      const response = await fetch(`${API_BASE}/api/assignment7/stereo`, {
        method: 'POST',
        body: fd,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.error ?? 'Request failed');
      }
      const payload = await response.json();
      setStereoResult(payload);
      setStereoStatus('Stereo measurement ready.');
    } catch (error: any) {
      setStereoStatus(error?.message ?? 'Stereo measurement failed.');
    }
  }, [leftFile, rightFile, vertices, scale]);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Assignment 7</p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Stereo Measurement + Pose &amp; Hand Tracking</h1>
          <p className="text-sm text-slate-300 sm:text-base">
            Recreate the calibrated stereo ruler and demonstrate Mediapipe-powered real-time pose + hand tracking with CSV logging.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <form onSubmit={runStereoMeasurement} className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-emerald-200">1. Calibrated Stereo Measurement</h2>
              <span className="text-xs text-slate-400">{stereoStatus}</span>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2 text-sm">
                <label className="text-slate-300">Left image</label>
                <input type="file" accept="image/*" onChange={handleLeftChange} className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
              </div>
              <div className="space-y-2 text-sm">
                <label className="text-slate-300">Right image</label>
                <input type="file" accept="image/*" onChange={handleRightChange} className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="text-xs text-slate-300">
                Baseline (mm)
                <input name="baselineMm" defaultValue="60" className="mt-1 w-full rounded border border-slate-700 bg-slate-900/60 px-2 py-1 text-sm" />
              </label>
              <label className="text-xs text-slate-300">
                Focal length (mm)
                <input name="focalMm" defaultValue="3.6" className="mt-1 w-full rounded border border-slate-700 bg-slate-900/60 px-2 py-1 text-sm" />
              </label>
              <label className="text-xs text-slate-300">
                Sensor width (mm)
                <input name="sensorWidthMm" defaultValue="3.2" className="mt-1 w-full rounded border border-slate-700 bg-slate-900/60 px-2 py-1 text-sm" />
              </label>
            </div>
            <canvas
              ref={drawingCanvasRef}
              onClick={handleCanvasClick}
              className="w-full max-w-full cursor-crosshair rounded border border-slate-800 bg-slate-900/60"
            />
            <div className="flex gap-2 text-xs text-slate-300">
              <button className="rounded border border-emerald-500/50 px-4 py-2 text-sm text-emerald-100 hover:bg-emerald-500/10" type="submit">
                Run Stereo Measurement
              </button>
              <button
                type="button"
                className="rounded border border-slate-600 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800/80"
                onClick={() => setVertices([])}
              >
                Clear Polygon
              </button>
            </div>
          </form>
          {stereoResult && (
            <div className="mt-4 space-y-4 text-sm text-slate-300">
              <figure className="space-y-2">
                <figcaption className="font-semibold text-slate-100">Disparity overview</figcaption>
                <img src={stereoResult.disparity} alt="Disparity map" className="rounded border border-slate-800" />
              </figure>
              <div className="rounded border border-slate-800 bg-slate-900/60 p-3">
                <p className="font-semibold text-slate-100">Segments</p>
                <ul className="mt-2 space-y-1 text-xs">
                  {stereoResult.segments.map((segment, index) => (
                    <li key={index} className="flex justify-between">
                      <span>
                        {index + 1}. [{segment.start.x.toFixed(1)}, {segment.start.y.toFixed(1)}] → [{segment.end.x.toFixed(1)}, {segment.end.y.toFixed(1)}]
                      </span>
                      <span className="text-emerald-300 font-semibold">{Number(segment.length_mm).toFixed(2)} mm</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-xs text-slate-400">Mean span: {Number(stereoResult.mean_length_mm).toFixed(2)} mm</p>
              </div>
            </div>
          )}
        </section>

        <PoseHandSection />
      </div>
    </div>
  );
}

function PoseHandSection() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const handRef = useRef<HandLandmarker | null>(null);
  const drawUtilsRef = useRef<DrawingUtils | null>(null);
  const animationRef = useRef<number | null>(null);
  const [status, setStatus] = useState<string>('Loading models...');
  const [csvRows, setCsvRows] = useState<CsvRow[]>([]);

  useEffect(() => {
    let mounted = true;
    const appendRows = (type: string, timestamp: number, landmarks: Array<{ x: number; y: number; z: number; visibility?: number }>) => {
      setCsvRows((prev) => {
        const next = [...prev];
        landmarks.forEach((landmark, index) => {
          next.push({ timestamp, type, index, x: landmark.x, y: landmark.y, z: landmark.z, visibility: landmark.visibility });
        });
        return next.slice(-2000);
      });
    };

    const startLoop = () => {
      const render = async () => {
        if (!videoRef.current || !canvasRef.current) {
          animationRef.current = requestAnimationFrame(render);
          return;
        }
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) {
          animationRef.current = requestAnimationFrame(render);
          return;
        }
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        const now = performance.now();
        if (poseRef.current) {
          const result = await poseRef.current.detectForVideo(videoRef.current, now);
          const poseLandmarks = result.landmarks?.[0];
          if (poseLandmarks && drawUtilsRef.current) {
            drawUtilsRef.current.drawLandmarks(poseLandmarks, { radius: 2, color: '#22d3ee' });
            drawUtilsRef.current.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#22d3ee' });
            appendRows('pose', now, poseLandmarks);
          }
        }
        if (handRef.current) {
          const result = await handRef.current.detectForVideo(videoRef.current, now);
          if (result.landmarks?.length && drawUtilsRef.current) {
            result.landmarks.forEach((landmarks, handIndex) => {
              drawUtilsRef.current?.drawLandmarks(landmarks, { radius: 2, color: '#f472b6' });
              drawUtilsRef.current?.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#f472b6' });
              appendRows(`hand${handIndex}`, now, landmarks);
            });
          }
        }
        animationRef.current = requestAnimationFrame(render);
      };
      animationRef.current = requestAnimationFrame(render);
    };

    (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm',
        );
        poseRef.current = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });
        handRef.current = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });
        setStatus('Models loaded. Requesting camera...');
        if (!mounted) return;
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false });
        if (!videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;
        drawUtilsRef.current = new DrawingUtils(ctx);
        setStatus('Tracking in progress...');
        startLoop();
      } catch (error) {
        console.error(error);
        setStatus('Failed to load Mediapipe assets.');
      }
    })();
    return () => {
      mounted = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((track) => track.stop());
      }
    };
  }, []);

  const downloadCsv = useCallback(() => {
    if (!csvRows.length) {
      alert('No data captured yet.');
      return;
    }
    const header = 'timestamp,type,index,x,y,z,visibility';
    const body = csvRows
      .map((row) => `${row.timestamp.toFixed(2)},${row.type},${row.index},${row.x.toFixed(5)},${row.y.toFixed(5)},${row.z.toFixed(5)},${row.visibility ?? ''}`)
      .join('\n');
    const blob = new Blob([`${header}\n${body}`], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'pose_hand_tracking.csv';
    link.click();
    URL.revokeObjectURL(url);
  }, [csvRows]);

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 text-sm text-slate-300">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-sky-200">3. Pose + Hand Tracking (Mediapipe)</h2>
        <span className="text-xs text-slate-400">{status}</span>
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <video ref={videoRef} muted playsInline className="hidden" />
        <canvas ref={canvasRef} width={640} height={360} className="w-full rounded border border-slate-800 bg-black" />
        <div className="space-y-3 text-xs text-slate-400">
          <p>Tracking data rows captured: {csvRows.length}</p>
          <button className="rounded border border-sky-500/50 px-4 py-2 text-sm text-sky-100 hover:bg-sky-500/10" onClick={downloadCsv} type="button">
            Download CSV
          </button>
          <p>
            The CSV logs timestamped 3D coordinates for every pose joint and hand landmark. Use it to annotate your report and explain what each column
            represents.
          </p>
        </div>
      </div>
    </section>
  );
}
