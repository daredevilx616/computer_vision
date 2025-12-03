'use client';

import React, { useCallback, useMemo, useState } from 'react';

type Status = 'idle' | 'processing' | 'done' | 'error';

type ApiResponse = {
  blurImage: string | null;
  restoredImage: string | null;
  montageImage: string | null;
  psnrBlur: number;
  psnrRestore: number;
  error?: string;
};

export default function FourierLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [message, setMessage] = useState<string>('Upload a photo taken with your camera and press Run.');
  const [blurImage, setBlurImage] = useState<string | null>(null);
  const [restoredImage, setRestoredImage] = useState<string | null>(null);
  const [montageImage, setMontageImage] = useState<string | null>(null);
  const [psnrBlur, setPsnrBlur] = useState<number | null>(null);
  const [psnrRestore, setPsnrRestore] = useState<number | null>(null);

  const reset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setBlurImage(null);
    setRestoredImage(null);
    setMontageImage(null);
    setPsnrBlur(null);
    setPsnrRestore(null);
    setStatus('idle');
    setMessage('Upload a photo taken with your camera and press Run.');
  }, [preview]);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (!selected) {
      reset();
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    const url = URL.createObjectURL(selected);
    setFile(selected);
    setPreview(url);
    setBlurImage(null);
    setRestoredImage(null);
    setMontageImage(null);
    setPsnrBlur(null);
    setPsnrRestore(null);
    setStatus('idle');
    setMessage('Ready. Click Run Fourier Pipeline to blur and restore.');
  }, [preview, reset]);

  const handleProcess = useCallback(async () => {
    if (!file) {
      setMessage('Please choose an image first.');
      return;
    }
    setStatus('processing');
    setMessage('Applying Gaussian blur and Fourier reconstruction …');
    try {
      const formData = new FormData();
      formData.append('image', file);
      const response = await fetch('/api/fourier', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload?.error ?? 'Server error');
      }
      const data: ApiResponse = await response.json();
      setBlurImage(data.blurImage);
      setRestoredImage(data.restoredImage);
      setMontageImage(data.montageImage);
      setPsnrBlur(typeof data.psnrBlur === 'number' ? data.psnrBlur : null);
      setPsnrRestore(typeof data.psnrRestore === 'number' ? data.psnrRestore : null);
      setStatus('done');
      setMessage('Complete! Compare the blurred and restored images below.');
    } catch (error: any) {
      console.error('Fourier processing failed', error);
      setStatus('error');
      setMessage(error?.message ?? 'Failed to process image.');
    }
  }, [file]);

  const statusMessage = useMemo(() => message, [message]);

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-4">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Module 2 · Fourier Blur Lab</h1>
          <p className="text-sm text-slate-300 sm:text-base">
            This experiment satisfies the “Gaussian blur then recover using Fourier transform” portion of Module&nbsp;2.
            Upload a photo you captured (mobile or webcam shot works), let the backend blur it with a {`${
              13
            }`}×{`${13}`} Gaussian kernel, and then attempt Wiener deblurring in the frequency domain. The results shown
            here match the command-line script in <code>module2/fourier_deblur.py</code>.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h2 className="text-lg font-medium">1. Upload &amp; Run</h2>
            <p className="text-sm text-slate-300">
              Pick an image captured by your own camera. Larger images are accepted but may take a few seconds.
            </p>
            <input
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleFileChange}
              className="w-full cursor-pointer rounded border border-slate-600 bg-slate-800 px-3 py-2 text-sm file:mr-4 file:rounded file:border-0 file:bg-slate-700 file:px-4 file:py-2"
            />
            <div className="flex gap-2 text-xs">
              <button
                onClick={handleProcess}
                disabled={!file || status === 'processing'}
                className="rounded border border-sky-400 bg-sky-500/20 px-4 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/30 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-transparent disabled:text-slate-500"
              >
                {status === 'processing' ? 'Processing…' : 'Run Fourier Pipeline'}
              </button>
              <button
                onClick={reset}
                className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/60 disabled:cursor-not-allowed"
                disabled={!file && !preview}
              >
                Reset
              </button>
            </div>
            <p className="text-xs text-slate-400">{statusMessage}</p>
            {preview && (
              <div className="rounded border border-slate-700 bg-slate-800/60 p-3 text-xs text-slate-300">
                <p className="mb-2 font-semibold text-slate-200">Original preview</p>
                <img src={preview} alt="Original upload preview" className="w-full rounded" />
              </div>
            )}
          </div>

          <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            <h2 className="text-lg font-medium">2. Results</h2>
            {status === 'processing' && <p className="text-sm text-slate-400">Working in the frequency domain…</p>}
            {status === 'error' && <p className="text-sm text-red-300">{statusMessage}</p>}
            {status !== 'processing' && !blurImage && (
              <p className="text-sm text-slate-400">
                Run the pipeline to see the Gaussian-blurred version, the Fourier reconstruction, and a side-by-side
                montage.
              </p>
            )}
            {blurImage && (
              <div className="grid gap-4 sm:grid-cols-3">
                <figure className="space-y-2">
                  <figcaption className="text-xs uppercase tracking-wide text-slate-400">Gaussian blur</figcaption>
                  <img src={blurImage} alt="Gaussian blur result" className="w-full rounded border border-slate-700" />
                </figure>
                <figure className="space-y-2">
                  <figcaption className="text-xs uppercase tracking-wide text-slate-400">Wiener restored</figcaption>
                  <img
                    src={restoredImage ?? ''}
                    alt="Fourier restored result"
                    className="w-full rounded border border-slate-700"
                  />
                </figure>
                <figure className="space-y-2 sm:col-span-3">
                  <figcaption className="text-xs uppercase tracking-wide text-slate-400">Montage</figcaption>
                  <img
                    src={montageImage ?? ''}
                    alt="Original vs blurred vs restored montage"
                    className="w-full rounded border border-slate-700"
                  />
                </figure>
              </div>
            )}
            {(psnrBlur !== null || psnrRestore !== null) && (
              <div className="rounded border border-slate-700 bg-slate-800/60 p-3 text-xs text-slate-300">
                <p className="font-semibold text-slate-100">Quality metrics (PSNR)</p>
                <p>Original → Blurred: {psnrBlur?.toFixed(2) ?? 'n/a'} dB</p>
                <p>Original → Restored: {psnrRestore?.toFixed(2) ?? 'n/a'} dB</p>
              </div>
            )}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-sm text-slate-300 md:text-base">
          <h2 className="text-lg font-semibold text-slate-100">Demo Video</h2>
          <p className="text-sm text-slate-400 mb-3">
            Quick walkthrough of the Gaussian blur + Fourier/Wiener deblur pipeline on real images.
          </p>
          <div className="w-full overflow-hidden rounded-xl border border-slate-700 bg-black aspect-video">
            <iframe
              className="w-full h-full"
              src="https://www.youtube.com/embed/hrnb7dWY5bw"
              title="Module 2 Fourier demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-sm text-slate-300 md:text-base">
          <h3 className="text-lg font-semibold text-slate-100">How it works</h3>
          <ol className="mt-3 space-y-2 list-decimal pl-6">
            <li>Convert your uploaded image to grayscale float values.</li>
            <li>Apply a {`${13}`}×{`${13}`} Gaussian blur (σ = 2.4) in the spatial domain.</li>
            <li>Build the same Gaussian as a point-spread function (PSF), shift it to the frequency origin, and compute its FFT.</li>
            <li>Use a Wiener filter (K = 1e-3) in the frequency domain to approximate the original image.</li>
            <li>Export PNGs for the blurred frame, the restored frame, and a side-by-side montage for reporting.</li>
          </ol>
        </section>
      </div>
    </div>
  );
}
