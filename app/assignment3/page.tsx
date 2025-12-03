'use client';

import React, { useCallback, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

async function postAssignment3(formData: FormData) {
  const response = await fetch(`${API_BASE}/api/assignment3`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload?.error ?? 'Request failed';
    throw new Error(message);
  }
  return response.json();
}

type ArucoResult = {
  overlay: string;
  mask: string;
  marker_count: number;
};

export default function Assignment3Page() {
  const [status, setStatus] = useState<string>('Idle');
  const [gradientResult, setGradientResult] = useState<{ magnitude: string; orientation: string; log: string } | null>(null);
  const [keypointOverlay, setKeypointOverlay] = useState<string | null>(null);
  const [boundaryResult, setBoundaryResult] = useState<{ overlay: string; mask: string } | null>(null);
  const [arucoResults, setArucoResults] = useState<ArucoResult[]>([]);
  const [compareMetrics, setCompareMetrics] = useState<{ dice: number; iou: number } | null>(null);
  const [samRefPreview, setSamRefPreview] = useState<string | null>(null);
  const [samCandPreview, setSamCandPreview] = useState<string | null>(null);

  const handleGradient = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem('gradientImage') as HTMLInputElement;
    if (!fileInput.files?.length) {
      alert('Upload an image first.');
      return;
    }
    const fd = new FormData();
    fd.append('operation', 'gradients');
    fd.append('image', fileInput.files[0]);
    setStatus('Computing gradients...');
    try {
      const result = await postAssignment3(fd);
      setGradientResult({
        magnitude: result.magnitude,
        orientation: result.orientation,
        log: result.log,
      });
      setStatus('Gradient analysis complete.');
    } catch (error: any) {
      setStatus(error?.message ?? 'Gradient analysis failed.');
    }
  }, []);

  const handleKeypoints = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem('keypointImage') as HTMLInputElement;
    const modeSelect = form.elements.namedItem('keypointMode') as HTMLSelectElement;
    if (!fileInput.files?.length) {
      alert('Upload an image first.');
      return;
    }
    const fd = new FormData();
    fd.append('operation', 'keypoints');
    fd.append('mode', modeSelect.value);
    fd.append('image', fileInput.files[0]);
    setStatus(`Detecting ${modeSelect.value === 'edge' ? 'edges' : 'corners'}...`);
    try {
      const result = await postAssignment3(fd);
      setKeypointOverlay(result.overlay);
      setStatus('Keypoints ready.');
    } catch (error: any) {
      setStatus(error?.message ?? 'Keypoint detection failed.');
    }
  }, []);

  const handleBoundary = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const input = (event.currentTarget.elements.namedItem('boundaryImage') as HTMLInputElement) ?? null;
    if (!input?.files?.length) {
      alert('Upload an image first.');
      return;
    }
    const fd = new FormData();
    fd.append('operation', 'boundary');
    fd.append('image', input.files[0]);
    setStatus('Locating dominant boundary...');
    try {
      const result = await postAssignment3(fd);
      setBoundaryResult({ overlay: result.overlay, mask: result.mask });
      setStatus('Boundary extracted.');
    } catch (error: any) {
      setStatus(error?.message ?? 'Boundary extraction failed.');
    }
  }, []);

  const handleAruco = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem('arucoImages') as HTMLInputElement;
    if (!input.files?.length) {
      alert('Select at least one image.');
      return;
    }
    const dictionary = (event.currentTarget.elements.namedItem('arucoDictionary') as HTMLSelectElement)?.value ?? 'DICT_5X5_100';
    setStatus('Processing ArUco batches...');
    const nextResults: ArucoResult[] = [];
    for (const file of Array.from(input.files)) {
      const fd = new FormData();
      fd.append('operation', 'aruco');
      fd.append('dictionary', dictionary);
      fd.append('image', file);
      try {
        const result = await postAssignment3(fd);
        nextResults.push({ overlay: result.overlay, mask: result.mask, marker_count: result.marker_count });
      } catch (error: any) {
        nextResults.push({ overlay: '', mask: '', marker_count: 0 });
        console.error('Aruco processing failed for', file.name, error);
      }
    }
    setArucoResults(nextResults);
    setStatus('ArUco segmentation complete.');
  }, []);

  const handleCompare = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const refInput = event.currentTarget.elements.namedItem('samReference') as HTMLInputElement;
    const candInput = event.currentTarget.elements.namedItem('samCandidate') as HTMLInputElement;
    if (!refInput.files?.length || !candInput.files?.length) {
      alert('Upload both masks.');
      return;
    }
    const fd = new FormData();
    fd.append('operation', 'compare');
    fd.append('reference', refInput.files[0]);
    fd.append('candidate', candInput.files[0]);
    setStatus('Comparing to SAM2 mask...');
    try {
      const result = await postAssignment3(fd);
      setCompareMetrics({ dice: result.dice, iou: result.iou });
      setStatus('Mask comparison complete.');
    } catch (error: any) {
      setStatus(error?.message ?? 'Comparison failed.');
    }
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-50">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-emerald-300">Assignment 3</p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Gradients, Keypoints, and Segmentation Lab</h1>
          <p className="text-sm text-slate-300 sm:text-base">
            Upload your dataset images to compute gradient &amp; LoG visualizations, detect edge/corner keypoints, recover the precise boundary,
            and evaluate ArUco-assisted segmentation against SAM2 masks.
          </p>
          <div className="text-xs text-slate-400">Status: {status}</div>
        </header>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <h2 className="text-lg font-semibold text-emerald-200">Module 3 Demo Video</h2>
          <div className="mt-3 space-y-2">
            <iframe
              className="w-full rounded border border-slate-800"
              style={{ aspectRatio: '16 / 9', minHeight: '315px' }}
              src="https://www.youtube.com/embed/dcgSvX3Yomo"
              title="Module 3 Demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
              frameBorder="0"
            />
            <a
              href="https://youtu.be/dcgSvX3Yomo"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-emerald-200 underline hover:text-emerald-100"
            >
              Open video in a new tab
            </a>
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <form onSubmit={handleGradient} className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="text-lg font-semibold text-emerald-200">1. Gradients &amp; LoG</h2>
            <input name="gradientImage" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
            <button className="rounded border border-emerald-500/50 px-4 py-2 text-sm text-emerald-200 hover:bg-emerald-500/10" type="submit">
              Compute Gradients
            </button>
            {gradientResult && (
              <div className="grid gap-3 text-xs text-slate-300 sm:grid-cols-3">
                <figure className="space-y-2">
                  <figcaption className="font-semibold text-slate-200">Magnitude</figcaption>
                  <img src={gradientResult.magnitude} alt="Gradient magnitude" className="rounded border border-slate-800" />
                </figure>
                <figure className="space-y-2">
                  <figcaption className="font-semibold text-slate-200">Orientation</figcaption>
                  <img src={gradientResult.orientation} alt="Gradient orientation" className="rounded border border-slate-800" />
                </figure>
                <figure className="space-y-2">
                  <figcaption className="font-semibold text-slate-200">LoG Filter</figcaption>
                  <img src={gradientResult.log} alt="LoG filtered" className="rounded border border-slate-800" />
                </figure>
              </div>
            )}
          </form>

          <form onSubmit={handleKeypoints} className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="text-lg font-semibold text-sky-200">2. Edge &amp; Corner Keypoints</h2>
            <input name="keypointImage" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
            <select name="keypointMode" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200">
              <option value="edge">Edge evidence</option>
              <option value="corner">Corner responses</option>
            </select>
            <button className="rounded border border-sky-500/50 px-4 py-2 text-sm text-sky-100 hover:bg-sky-500/10" type="submit">
              Detect Keypoints
            </button>
            {keypointOverlay && (
              <figure className="space-y-2 text-xs text-slate-300">
                <figcaption className="font-semibold text-slate-200">Overlay</figcaption>
                <img src={keypointOverlay} alt="Keypoint overlay" className="rounded border border-slate-800" />
              </figure>
            )}
          </form>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <form onSubmit={handleBoundary} className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="text-lg font-semibold text-purple-200">3. Exact Object Boundary</h2>
            <input name="boundaryImage" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
            <button className="rounded border border-purple-500/50 px-4 py-2 text-sm text-purple-100 hover:bg-purple-500/10" type="submit">
              Segment Boundary
            </button>
            {boundaryResult && (
              <div className="grid gap-4 text-xs text-slate-300 sm:grid-cols-2">
                <figure className="space-y-2">
                  <figcaption className="font-semibold text-slate-100">Overlay</figcaption>
                  <img src={boundaryResult.overlay} alt="Boundary overlay" className="rounded border border-slate-800" />
                </figure>
                <figure className="space-y-2">
                  <figcaption className="font-semibold text-slate-100">Mask</figcaption>
                  <img src={boundaryResult.mask} alt="Boundary mask" className="rounded border border-slate-800" />
                </figure>
              </div>
            )}
          </form>

          <form onSubmit={handleAruco} className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="text-lg font-semibold text-amber-200">4. ArUco Segmentation (&gt;=10 shots)</h2>
            <input multiple name="arucoImages" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
            <select name="arucoDictionary" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200">
              <option value="DICT_5X5_100">DICT_5X5_100</option>
              <option value="DICT_4X4_50">DICT_4X4_50</option>
              <option value="DICT_APRILTAG_36h11">DICT_APRILTAG_36h11</option>
            </select>
            <button className="rounded border border-amber-500/50 px-4 py-2 text-sm text-amber-100 hover:bg-amber-500/10" type="submit">
              Process Batch
            </button>
            {arucoResults.length > 0 && (
              <div className="space-y-4 text-xs text-slate-300">
                {arucoResults.map((result, index) => (
                  <div key={index} className="rounded border border-slate-800 p-3">
                    <p className="font-semibold text-slate-100">Image {index + 1}</p>
                    <p className="text-slate-400">Detected markers: {result.marker_count}</p>
                    {result.overlay && <img src={result.overlay} alt={`Aruco overlay ${index + 1}`} className="mt-2 rounded border border-slate-800" />}
                  </div>
                ))}
              </div>
            )}
          </form>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <form onSubmit={handleCompare} className="space-y-3">
            <h2 className="text-lg font-semibold text-rose-200">5. Compare Against SAM2 Masks</h2>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2 text-sm">
                <label className="text-slate-300">Reference mask (SAM2)</label>
                <input
                  name="samReference"
                  type="file"
                  accept="image/*"
                  className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    setSamRefPreview(file ? URL.createObjectURL(file) : null);
                  }}
                />
              </div>
              <div className="space-y-2 text-sm">
                <label className="text-slate-300">Our segmentation mask</label>
                <input
                  name="samCandidate"
                  type="file"
                  accept="image/*"
                  className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    setSamCandPreview(file ? URL.createObjectURL(file) : null);
                  }}
                />
              </div>
            </div>
            <button className="rounded border border-rose-500/50 px-4 py-2 text-sm text-rose-100 hover:bg-rose-500/10" type="submit">
              Evaluate Masks
            </button>
            <div className="grid gap-3 pt-2 text-xs text-slate-300 md:grid-cols-2">
              {samRefPreview && (
                <figure className="space-y-1">
                  <figcaption className="font-semibold text-slate-200">Reference (SAM2)</figcaption>
                  <img src={samRefPreview} alt="SAM2 reference preview" className="rounded border border-slate-800" />
                </figure>
              )}
              {samCandPreview && (
                <figure className="space-y-1">
                  <figcaption className="font-semibold text-slate-200">Our mask</figcaption>
                  <img src={samCandPreview} alt="Our mask preview" className="rounded border border-slate-800" />
                </figure>
              )}
            </div>
          </form>
          {compareMetrics && (
            <div className="mt-4 rounded border border-slate-800 bg-slate-900/60 p-3 text-sm text-slate-200">
              <p>Dice coefficient: {compareMetrics.dice.toFixed(3)}</p>
              <p>IoU: {compareMetrics.iou.toFixed(3)}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
