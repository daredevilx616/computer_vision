'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/measure", label: "Module 1 · Measurement" },
  { href: "/template-matching", label: "Module 2 · Templates" },
  { href: "/fourier", label: "Module 2 · Fourier" },
  { href: "/assignment3", label: "Module 3" },
  { href: "/assignment4", label: "Module 4" },
  { href: "/assignment56", label: "Modules 5-6" },
  { href: "/assignment7", label: "Module 7" },
];

export default function Navbar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-4">
        <Link href="/" className="text-base font-semibold tracking-wide text-emerald-200">
          CV Toolkit
        </Link>
        <nav className="flex flex-wrap items-center gap-2 text-xs font-medium sm:text-sm">
          {navItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-full border px-3 py-1 transition ${
                  active
                    ? "border-emerald-400 text-emerald-200"
                    : "border-slate-700/60 text-slate-300 hover:border-emerald-400/70 hover:text-emerald-200"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
