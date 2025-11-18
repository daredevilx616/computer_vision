'use client';

import React, { useCallback, useState } from 'react';

async function postForm(url: string, formData: FormData) {
  const response = await fetch(url, { method: 'POST', body: formData });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload?.error ?? 'Request failed');
  }
  return response.json();
}

export default function Assignment4Page() {
  const [siftStatus, setSiftStatus] = useState<string>('Idle');
  const [stitchStatus, setStitchStatus] = useState<string>('Idle');
  const [siftResult, setSiftResult] = useState<{ match_count: number; inliers: number; homography: number[][]; visual?: string } | null>(null);
  const [panoramaResult, setPanoramaResult] = useState<{ panorama: string; match_visuals: string[] } | null>(null);

  const handleSift = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const a = form.elements.namedItem('siftA') as HTMLInputElement;
    const b = form.elements.namedItem('siftB') as HTMLInputElement;
    if (!a.files?.length || !b.files?.length) {
      alert('Upload both image A and image B.');
      return;
    }
    const fd = new FormData();
    fd.append('imageA', a.files[0]);
    fd.append('imageB', b.files[0]);
    setSiftStatus('Running custom SIFT + RANSAC...');
    try {
      const payload = await postForm('/api/assignment4/sift', fd);
      setSiftResult(payload);
      setSiftStatus('SIFT comparison ready.');
    } catch (error: any) {
      setSiftStatus(error?.message ?? 'SIFT run failed.');
    }
  }, []);

  const handleStitch = useCallback(async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem('stitchImages') as HTMLInputElement;
    if (!input.files || input.files.length < 2) {
      alert('Select at least two images');
      return;
    }
    const fd = new FormData();
    Array.from(input.files).forEach((file) => fd.append('images', file));
    setStitchStatus('Generating panorama...');
    try {
      const payload = await postForm('/api/assignment4/stitch', fd);
      setPanoramaResult(payload);
      setStitchStatus('Panorama complete.');
    } catch (error: any) {
      setStitchStatus(error?.message ?? 'Panorama failed.');
    }
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-10">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Assignment 4</p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">SIFT from Scratch + Panorama Stitching</h1>
          <p className="text-sm text-slate-300 sm:text-base">
            Demonstrate the handcrafted SIFT pipeline with RANSAC homography estimation and compare it with a full panorama stitch that merges at least
            four perspectives.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <form onSubmit={handleSift} className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-emerald-200">Custom SIFT + RANSAC</h2>
              <span className="text-xs text-slate-400">{siftStatus}</span>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-1 text-sm">
                <label className="text-slate-300">Image A</label>
                <input name="siftA" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
              </div>
              <div className="space-y-1 text-sm">
                <label className="text-slate-300">Image B</label>
                <input name="siftB" type="file" accept="image/*" className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm" />
              </div>
            </div>
            <button className="rounded border border-emerald-500/50 px-4 py-2 text-sm text-emerald-100 hover:bg-emerald-500/10" type="submit">
              Run SIFT Pipeline
            </button>
          </form>
          {siftResult && (
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <p>
                Matches: <span className="font-semibold text-slate-100">{siftResult.match_count}</span> · Inliers:{' '}
                <span className="font-semibold text-slate-100">{siftResult.inliers}</span>
              </p>
              <div className="overflow-auto rounded border border-slate-800 bg-slate-900/60 p-2">
                <pre className="text-xs leading-snug text-slate-400">{JSON.stringify(siftResult.homography, null, 2)}</pre>
              </div>
              {siftResult.visual && <img src={siftResult.visual} alt="SIFT matches" className="rounded border border-slate-800" />}
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <form onSubmit={handleStitch} className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-sky-200">Panorama Stitcher (&gt;=4 landscape frames)</h2>
              <span className="text-xs text-slate-400">{stitchStatus}</span>
            </div>
            <input
              name="stitchImages"
              multiple
              type="file"
              accept="image/*"
              className="w-full rounded border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm"
            />
            <button className="rounded border border-sky-500/50 px-4 py-2 text-sm text-sky-100 hover:bg-sky-500/10" type="submit">
              Stitch Images
            </button>
          </form>
          {panoramaResult && (
            <div className="mt-4 space-y-4 text-sm text-slate-300">
              <div>
                <p className="font-semibold text-slate-100">Panorama</p>
                <img src={panoramaResult.panorama} alt="Panorama" className="mt-2 rounded border border-slate-800" />
              </div>
              {panoramaResult.match_visuals?.length ? (
                <div className="space-y-2">
                  <p className="font-semibold text-slate-100">Pairwise matches</p>
                  {panoramaResult.match_visuals.map((visual, index) => (
                    <img key={index} src={visual} alt={`Match ${index}`} className="rounded border border-slate-800" />
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
