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
  const [thresholdValue, setThresholdValue] = useState<number>(0.7);

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

  const getColorForLabel = useCallback((label: string, index: number) => {
    // Define distinct colors for different labels
    const colorMap: { [key: string]: { stroke: string; fill: string; text: string } } = {
      heart: { stroke: 'rgba(255, 75, 95, 0.95)', fill: '#ff4b5f', text: '#ffb3bd' },
      diamond: { stroke: 'rgba(75, 150, 255, 0.95)', fill: '#4b96ff', text: '#b3d7ff' },
      club: { stroke: 'rgba(150, 255, 75, 0.95)', fill: '#96ff4b', text: '#d7ffb3' },
      spade: { stroke: 'rgba(180, 100, 255, 0.95)', fill: '#b464ff', text: '#ddb3ff' },
    };

    // Try to match by label
    if (colorMap[label]) return colorMap[label];

    // Fallback to index-based colors
    const fallbackColors = [
      { stroke: 'rgba(45, 255, 196, 0.95)', fill: '#2dffc4', text: '#8fffd1' },
      { stroke: 'rgba(255, 196, 45, 0.95)', fill: '#ffc42d', text: '#ffd98f' },
      { stroke: 'rgba(255, 45, 196, 0.95)', fill: '#ff2dc4', text: '#ff8fd9' },
      { stroke: 'rgba(196, 45, 255, 0.95)', fill: '#c42dff', text: '#d98fff' },
    ];

    return fallbackColors[index % fallbackColors.length];
  }, []);

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

        detectionList.forEach((det, index) => {
          const { left, top, right, bottom } = det.box;
          const width = right - left;
          const height = bottom - top;
          const colors = getColorForLabel(det.label, index);

          ctx.save();
          ctx.beginPath();
          ctx.rect(left, top, width, height);
          ctx.clip();
          ctx.filter = 'blur(10px)';
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          ctx.restore();

          ctx.save();
          ctx.strokeStyle = colors.stroke;
          ctx.lineWidth = 3;
          ctx.strokeRect(left, top, width, height);
          ctx.fillStyle = 'rgba(12, 12, 12, 0.7)';
          ctx.fillRect(left, Math.max(0, top - 20), width, 20);
          ctx.fillStyle = colors.text;
          ctx.font = '14px "Segoe UI", sans-serif';
          const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
          ctx.fillText(label, left + 6, Math.max(14, top - 6));
          ctx.restore();
        });
      };
      img.src = imageUrl;
    },
    [getColorForLabel],
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
      formData.append('threshold', thresholdValue.toString());
      const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';
      const response = await fetch(`${API_BASE}/api/template-matching`, {
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
  }, [sceneFile, scenePreview, drawBlurredDetections, thresholdValue]);

  const statusMessage = useMemo(() => {
    return message;
  }, [message]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 px-6 py-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Module 2 &mdash; Template Matching Sandbox</h1>
          <p className="text-sm text-slate-300">
            Upload a scene image to detect ALL available templates (hearts, diamonds, clubs, spades, logos, etc.).
            The system runs multi-scale template matching and highlights all detected objects with unique colors.
            The backend uses OpenCV correlation and calls Python scripts in <code>module2/</code>.
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
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Detection Threshold: {thresholdValue.toFixed(2)} ({(thresholdValue * 100).toFixed(0)}%)
                </label>
                <input
                  type="range"
                  min="0.3"
                  max="0.95"
                  step="0.05"
                  value={thresholdValue}
                  onChange={(e) => setThresholdValue(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>0.3 (More, may have false positives)</span>
                  <span>0.95 (Only very confident matches)</span>
                </div>
              </div>
            </div>
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
                {detections.map((item, index) => {
                  const colors = getColorForLabel(item.label, index);
                  return (
                    <li
                      key={`${item.label}-${item.box.left}-${item.box.top}`}
                      className="rounded border p-3"
                      style={{
                        borderColor: colors.stroke.replace('0.95', '0.3'),
                        backgroundColor: colors.stroke.replace('0.95', '0.1'),
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold" style={{ color: colors.text }}>
                          {item.label}
                        </span>
                        <span>{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="text-xs opacity-80">
                        Box: {item.box.left}, {item.box.top} &rarr; {item.box.right}, {item.box.bottom}
                      </div>
                    </li>
                  );
                })}
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
          <h2 className="text-lg font-medium">Demo Video</h2>
          <p className="text-sm text-slate-300 mb-3">
            Quick walkthrough of uploading scenes, adjusting threshold, detecting objects, and blurring detected regions.
          </p>
          <div className="w-full overflow-hidden rounded-xl border border-slate-700 bg-black aspect-video">
            <iframe
              className="w-full h-full"
              src="https://www.youtube.com/embed/sfWhuLJTlMk"
              title="Module 2 template demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-5">
          <h2 className="text-lg font-medium">Features</h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            <li>✓ Multi-template detection - detects ALL available templates in a single scene.</li>
            <li>✓ Multi-scale matching - finds objects at different sizes automatically.</li>
            <li>✓ Unique color coding - each object type has its own distinctive color.</li>
            <li>✓ API endpoint powering this UI (`POST /api/template-matching`).</li>
            <li>✓ Browser-side blur overlay with bounding boxes for each detection.</li>
            <li>✓ Confidence scores and precise bounding box coordinates displayed.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
