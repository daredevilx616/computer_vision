'use client';

import React, { useCallback, useMemo, useRef, useState } from 'react';

type Detection = {
  label: string;
  confidence: number;
  box: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
};

type ApiResponse = {
  detections: Detection[];
  threshold: number;
  annotatedImage: string | null;
  sourceScene: string;
  error?: string;
};

type Status = 'idle' | 'processing' | 'done' | 'error';

export default function TemplateMatchingPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [sceneFile, setSceneFile] = useState<File | null>(null);
  const [scenePreview, setScenePreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [message, setMessage] = useState<string>('Upload a scene and press Process to run detection.');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);

  const resetPreview = useCallback(() => {
    if (scenePreview) {
      URL.revokeObjectURL(scenePreview);
    }
    setScenePreview(null);
    setSceneFile(null);
    setDetections([]);
    setAnnotatedImage(null);
    setThreshold(null);
    setStatus('idle');
    setMessage('Upload a scene and press Process to run detection.');
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [scenePreview]);

  const drawBlurredDetections = useCallback(
    (imageUrl: string, detectionList: Detection[]) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        detectionList.forEach((det) => {
          const { left, top, right, bottom } = det.box;
          const width = right - left;
          const height = bottom - top;
          ctx.save();
          ctx.beginPath();
          ctx.rect(left, top, width, height);
          ctx.clip();
          ctx.filter = 'blur(10px)';
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          ctx.save();
          ctx.strokeStyle = 'rgba(45, 255, 196, 0.95)';
          ctx.lineWidth = 3;
          ctx.strokeRect(left, top, width, height);
          ctx.fillStyle = 'rgba(12, 12, 12, 0.7)';
          ctx.fillRect(left, Math.max(0, top - 20), width, 20);
          ctx.fillStyle = '#8fffd1';
          ctx.font = '14px "Segoe UI", sans-serif';
          const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
          ctx.fillText(label, left + 6, Math.max(14, top - 6));
          ctx.restore();
        });
      };
      img.src = imageUrl;
    },
    [],
  );

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) {
        resetPreview();
        return;
      }
      if (scenePreview) URL.revokeObjectURL(scenePreview);
      const url = URL.createObjectURL(file);
      setSceneFile(file);
      setScenePreview(url);
      setDetections([]);
      setAnnotatedImage(null);
      setThreshold(null);
      setStatus('idle');
      setMessage('Ready to process.');
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const img = new Image();
          img.onload = () => {
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          };
          img.src = url;
        }
      }
    },
    [resetPreview, scenePreview],
  );

  const handleProcessClick = useCallback(async () => {
    if (!sceneFile || !scenePreview) {
      setMessage('Please choose an image first.');
      return;
    }
    setStatus('processing');
    setMessage('Running correlation on uploaded scene …');
    try {
      const formData = new FormData();
      formData.append('scene', sceneFile);
      const response = await fetch('/api/template-matching', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorPayload = await response.json();
        throw new Error(errorPayload?.error ?? 'Server error');
      }
      const data: ApiResponse = await response.json();
      const detectedThreshold = typeof data.threshold === 'number' ? data.threshold : null;
      setDetections(data.detections ?? []);
      setThreshold(detectedThreshold);
      setAnnotatedImage(data.annotatedImage ?? null);
      drawBlurredDetections(scenePreview, data.detections ?? []);
      if ((data.detections ?? []).length === 0) {
        setMessage('No templates detected above threshold.');
      } else {
        const prettyThreshold = detectedThreshold !== null ? detectedThreshold.toFixed(2) : 'n/a';
        setMessage(`Detected ${data.detections.length} template(s) at threshold ${prettyThreshold}.`);
      }
      setStatus('done');
    } catch (error: any) {
      console.error('Processing failed', error);
      setStatus('error');
      setMessage(error?.message ?? 'Something went wrong while processing.');
    }
  }, [sceneFile, scenePreview, drawBlurredDetections]);

  const statusMessage = useMemo(() => {
    return message;
  }, [message]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 px-6 py-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Module 2 &mdash; Template Matching Sandbox</h1>
          <p className="text-sm text-slate-300">
            Upload a scene, run the OpenCV correlation pipeline, and preview the blurred detections. The backend calls
            into the Python scripts in <code>module2/</code> so the web app now reflects the same results as the CLI.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
            <h2 className="text-lg font-medium">1. Upload Scene</h2>
            <p className="text-sm text-slate-300">Provide an RGB scene image (PNG/JPG). Mobile capture works.</p>
            <input
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
              className="w-full cursor-pointer rounded border border-slate-600 bg-slate-800 px-3 py-2 text-sm file:mr-4 file:rounded file:border-0 file:bg-slate-700 file:px-4 file:py-2"
            />
            <div className="flex gap-2 text-xs">
              <button
                onClick={handleProcessClick}
                disabled={!sceneFile || status === 'processing'}
                className="rounded border border-emerald-400 bg-emerald-500/20 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-500/30 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-transparent disabled:text-slate-500"
              >
                {status === 'processing' ? 'Processing…' : 'Process Scene'}
              </button>
              <button
                onClick={resetPreview}
                className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/60 disabled:cursor-not-allowed"
                disabled={!sceneFile && !scenePreview}
              >
                Reset
              </button>
            </div>
            <p className="text-xs text-slate-400">{statusMessage}</p>
            <div className="rounded border border-slate-700 bg-slate-800/60">
              <canvas ref={canvasRef} className="w-full rounded" />
            </div>
          </div>

          <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
            <h2 className="text-lg font-medium">2. Detected Templates</h2>
            {status === 'processing' && (
              <p className="text-sm text-slate-400">Crunching numbers&hellip;</p>
            )}
            {status === 'error' && (
              <p className="text-sm text-red-300">Processing failed. {statusMessage}</p>
            )}
            {detections.length === 0 && status === 'done' && (
              <p className="text-sm text-slate-400">No detections above threshold. Try another view or adjust lighting.</p>
            )}
            {detections.length === 0 && status !== 'done' && status !== 'error' && (
              <p className="text-sm text-slate-400">No detections yet. Upload and process a scene to view results.</p>
            )}
            {detections.length > 0 && (
              <ul className="space-y-3 text-sm">
                {detections.map((item) => (
                  <li key={`${item.label}-${item.box.left}-${item.box.top}`} className="rounded border border-emerald-400/30 bg-emerald-400/10 p-3">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-emerald-200">{item.label}</span>
                      <span>{(item.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-xs text-emerald-200/80">
                      Box: {item.box.left}, {item.box.top} &rarr; {item.box.right}, {item.box.bottom}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            {annotatedImage && (
              <div className="space-y-2 text-xs text-slate-300">
                <p>Server-side annotated preview:</p>
                <img src={annotatedImage} alt="Annotated detections" className="w-full rounded border border-slate-700" />
              </div>
            )}
            {threshold !== null && (
              <div className="text-xs text-slate-400">Threshold: {threshold.toFixed(2)}</div>
            )}
          </div>
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-5">
          <h2 className="text-lg font-medium">Build Checklist</h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            <li>✓ Synthetic dataset + CLI pipeline (`python -m module2.run_template_matching`).</li>
            <li>✓ Fourier blur/deblur experiment (`python -m module2.fourier_deblur`).</li>
            <li>✓ API endpoint powering this UI (`POST /api/template-matching`).</li>
            <li>✓ Browser-side blur overlay based on detection boxes.</li>
            <li>⏳ Add user controls (e.g., adjustable threshold, download results).</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
