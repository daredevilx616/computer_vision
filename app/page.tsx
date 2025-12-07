import Link from "next/link";

const moduleLinks = [
  { href: "/measure", label: "Module 1 · Measurement App", border: "border-emerald-400/70", bg: "bg-emerald-500/15", hover: "hover:bg-emerald-500/30" },
  { href: "/template-matching", label: "Module 2 · Template Matching", border: "border-sky-400/70", bg: "bg-sky-500/15", hover: "hover:bg-sky-500/30" },
  { href: "/fourier", label: "Module 2 · Fourier Lab", border: "border-purple-400/70", bg: "bg-purple-500/15", hover: "hover:bg-purple-500/30" },
  { href: "/assignment3", label: "Module 3 · Gradients & Segmentation", border: "border-emerald-400/70", bg: "bg-emerald-500/10", hover: "hover:bg-emerald-500/20" },
  { href: "/assignment4", label: "Module 4 · SIFT & Panorama", border: "border-sky-400/70", bg: "bg-sky-500/10", hover: "hover:bg-sky-500/20" },
  { href: "/assignment56", label: "Module 5-6 · Tracking Suite", border: "border-amber-400/70", bg: "bg-amber-500/10", hover: "hover:bg-amber-500/20" },
  { href: "/assignment7", label: "Module 7 · Stereo + Pose", border: "border-purple-400/70", bg: "bg-purple-500/10", hover: "hover:bg-purple-500/20" },
];

const highlightCards = [
  {
    title: "Module 1 · Measurement",
    accent: "text-emerald-200",
    border: "border-emerald-500/40 shadow-emerald-500/5",
    points: [
      "Live camera capture with annotation of measurement points.",
      "Perspective projection equations for millimetre-scale estimates.",
      "Seed experiment script available in measure.py for reports.",
    ],
  },
  {
    title: "Module 2 · Templates",
    accent: "text-sky-200",
    border: "border-sky-500/40 shadow-sky-500/5",
    points: [
      "Synthetic template dataset and OpenCV correlation pipeline.",
      "Blur-on-hit rendering driven by the Python detector API.",
      "Batch evaluation CLI writes annotated PNGs plus precision metrics.",
    ],
  },
  {
    title: "Module 2 · Fourier Lab",
    accent: "text-purple-200",
    border: "border-purple-500/40 shadow-purple-500/5",
    points: [
      "Upload your own camera photo to generate blurred/restored results.",
      "Wiener filter implementation mirrors the CLI workflow for reports.",
      "PSNR metrics surfaced directly in the browser for quick documentation.",
    ],
  },
  {
    title: "Module 3 · Gradients & Segmentation",
    accent: "text-emerald-200",
    border: "border-emerald-500/40 shadow-emerald-500/5",
    points: [
      "Compute gradient, orientation, and LoG maps from uploaded images.",
      "Edge & corner keypoint overlays plus ArUco-assisted segmentation.",
      "Dice/IoU comparison utilities to benchmark against SAM2 masks.",
    ],
  },
  {
    title: "Module 4 · SIFT + Panoramas",
    accent: "text-sky-200",
    border: "border-sky-500/40 shadow-sky-500/5",
    points: [
      "Scratch SIFT pipeline with descriptor matching and RANSAC homography.",
      "Match visualizations rendered as part of the web workflow.",
      "Panorama stitcher for ≥4 frames to mirror the assignment brief.",
    ],
  },
  {
    title: "Modules 5-6 · Tracking Suite",
    accent: "text-amber-200",
    border: "border-amber-500/40 shadow-amber-500/5",
    points: [
      "js-aruco fiducial detection, markerless color tracker, and SAM2 seeding.",
      "All trackers run simultaneously from the same camera feed.",
      "Designed for live demo capture—no additional tooling required.",
    ],
  },
  {
    title: "Module 7 · Stereo + Pose",
    accent: "text-purple-200",
    border: "border-purple-500/40 shadow-purple-500/5",
    points: [
      "Upload stereo pairs, draw polygons, and recover metric edge lengths.",
      "Disparity map visualization alongside computed measurements.",
      "Mediapipe pose + hand tracking with downloadable CSV logs.",
    ],
  },
];

export default function Home() {
  return (
    <main className="bg-slate-950">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 pb-20 pt-12">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 text-sm text-slate-300 md:text-base">
          <h2 className="text-lg font-semibold text-slate-100">How to get started</h2>
          <ol className="mt-3 space-y-2 list-decimal pl-6">
            <li>
              Install dependencies once with <code>python -m pip install -r requirements.txt</code>.
            </li>
            <li>
              Launch the dev server via <code>npm run dev</code> and open this homepage.
            </li>
            <li>
              Pick any module below to run the interactive workflow and capture results for your report or video submission.
            </li>
            <li>Use the Fourier Lab with an image from your camera to satisfy the blur/deblur requirement.</li>
          </ol>
        </section>

        <section className="space-y-4 text-slate-200">
          <p className="max-w-3xl text-base text-slate-300 sm:text-lg">
            Launch the individual assignment demos directly from here. Each button routes to a page connected to its Python backend so you can upload
            data, run the experiment, and capture screenshots without juggling scripts.
          </p>
          <p className="text-xs text-slate-400">
            GitHub repo:{' '}
            <a href="https://github.com/daredevilx616/computer_vision" className="text-blue-300 underline" target="_blank" rel="noreferrer">
              github.com/daredevilx616/computer_vision
            </a>
          </p>
          <div className="flex flex-wrap gap-3">
            {moduleLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full border ${link.border} ${link.bg} px-4 py-2 text-xs font-medium text-slate-100 transition ${link.hover}`}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          {highlightCards.map((card) => (
            <article
              key={card.title}
              className={`rounded-2xl border ${card.border} bg-slate-900/60 p-6 text-sm text-slate-300 shadow-lg`}
            >
              <h3 className={`text-lg font-semibold ${card.accent}`}>{card.title}</h3>
              <ul className="mt-4 space-y-3">
                {card.points.map((point) => (
                  <li key={point} className="pl-4 text-slate-300">
                    {point}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}
