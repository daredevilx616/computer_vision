import Link from "next/link";
import Navbar from "../components/Navbar";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-50">
      <Navbar />
      <section className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-16 lg:py-20">
        <header className="space-y-6">
          <span className="inline-flex items-center rounded-full border border-emerald-400/40 bg-emerald-400/10 px-4 py-1 text-xs uppercase tracking-[0.3em] text-emerald-200">
            Computer Vision Toolkit
          </span>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
            Measure the real world and detect templates right in your browser.
          </h1>
          <p className="max-w-2xl text-base text-slate-300 sm:text-lg">
            Explore two interactive modules: a perspective-based ruler for Module&nbsp;1 and an
            object detector with blur-on-hit for Module&nbsp;2. Each demo connects to the Python
            pipelines under the hood so your experiments stay reproducible.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/measure"
              className="rounded-lg border border-emerald-400 bg-emerald-500/20 px-5 py-3 text-sm font-medium text-emerald-100 transition hover:bg-emerald-500/30"
            >
              Module 1 · Measurement App
            </Link>
            <Link
              href="/template-matching"
              className="rounded-lg border border-sky-400 bg-sky-500/20 px-5 py-3 text-sm font-medium text-sky-100 transition hover:bg-sky-500/30"
            >
              Module 2 · Template Matching
            </Link>
            <Link
              href="/fourier"
              className="rounded-lg border border-purple-400 bg-purple-500/20 px-5 py-3 text-sm font-medium text-purple-100 transition hover:bg-purple-500/30"
            >
              Module 2 · Fourier Lab
            </Link>
          </div>
        </header>

        <section className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-emerald-500/5">
            <h2 className="text-xl font-semibold text-emerald-200">Module 1 Highlights</h2>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li>• Live camera capture with annotation of measurement points.</li>
              <li>• Perspective projection equations for millimetre-scale estimates.</li>
              <li>• Seed experiment script available in <code>measure.py</code> for reports.</li>
            </ul>
          </article>

          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-sky-500/5">
            <h2 className="text-xl font-semibold text-sky-200">Module 2 Highlights</h2>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li>• Synthetic template dataset and OpenCV correlation pipeline.</li>
              <li>• Fourier-domain blur &amp; deblur experiment with saved outputs.</li>
              <li>• Web UI that blurs detected regions using Python-backed API.</li>
            </ul>
          </article>

          <article className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-purple-500/5">
            <h2 className="text-xl font-semibold text-purple-200">Fourier Lab Highlights</h2>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li>• Upload your own camera photo to generate blurred/restored results.</li>
              <li>• Wiener filter implementation mirrors the CLI workflow for reports.</li>
              <li>• PSNR metrics surfaced directly in the browser for quick documentation.</li>
            </ul>
          </article>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 text-sm text-slate-300 md:text-base">
          <h3 className="text-lg font-semibold text-slate-100">How to get started</h3>
          <ol className="mt-3 space-y-2 list-decimal pl-6">
            <li>Install dependencies once with <code>python -m pip install -r requirements.txt</code>.</li>
            <li>Launch the dev server via <code>npm run dev</code> and open this homepage.</li>
            <li>Pick a module above to try the interactive workflow, then capture results for your report or video submission.</li>
            <li>Use the Fourier Lab with an image from your camera to satisfy the blur/deblur requirement.</li>
          </ol>
        </section>
      </section>
    </main>
  );
}
