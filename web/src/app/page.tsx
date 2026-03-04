"use client";

import Nav from "@/components/nav";
import Waitlist from "@/components/waitlist";

const ACCENT = "#22C55E";
const HUB_URL = "https://specialized-model-startups.vercel.app";


function SectionLabel({ label }: { label: string }) {
  return (
    <div className="reveal flex items-center gap-5 mb-12">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-[#0a0a0a] overflow-x-hidden">
      <Nav />

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 pt-14 overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 30%, ${ACCENT}07 0%, transparent 50%), radial-gradient(circle at 80% 70%, ${ACCENT}05 0%, transparent 50%)`,
          }}
        />

        <div className="relative max-w-5xl mx-auto w-full py-20">
          <div className="fade-up delay-0 mb-8">
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ color: ACCENT, borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}08` }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
              Training &middot; 18&times; A6000 &middot; ETA Q3 2026
            </span>
          </div>

          <h1 className="fade-up delay-1 text-[clamp(3rem,9vw,6.5rem)] font-bold leading-[0.92] tracking-tight mb-6">
            <span className="serif font-light italic" style={{ color: ACCENT }}>Green</span>
            <span>light CI</span>
          </h1>

          <p className="fade-up delay-2 serif text-[clamp(1.25rem,3vw,2rem)] font-light text-gray-500 mb-4 max-w-xl">
            CI never stays broken.
          </p>

          <p className="fade-up delay-3 text-sm text-gray-400 leading-relaxed max-w-lg mb-10">
            First model trained on CI failure taxonomy &times; minimal-diff &times; rerun-stability triplet&nbsp;— not just &ldquo;fix the error&rdquo; but &ldquo;fix it safely, small, and permanently.&rdquo;
          </p>

          <div className="fade-up delay-4">
            <Waitlist />
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="The Problem" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="reveal rounded-2xl border border-gray-100 p-8 bg-gray-50/50">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-5">What general models do</p>
            <ul className="space-y-3 text-sm text-gray-500 leading-relaxed">
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Fix one error, introduce two new ones
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Generate 500-line diffs that reviewers reject on principle
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Don&apos;t verify fixes against rerun — same failure next push
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                No understanding of CI taxonomy: flaky vs. dep drift vs. env
              </li>
            </ul>
          </div>

          <div
            className="reveal rounded-2xl border p-8"
            style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest mb-5" style={{ color: ACCENT }}>What Greenlight CI does</p>
            <ul className="space-y-3 text-sm leading-relaxed text-gray-700">
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Constrained by three hard rules: CI must pass, diff must be minimal, fix must hold on rerun
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Classifies failures before fixing: flaky, dep drift, or env-specific
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Verifies every fix on rerun before declaring it stable
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Produces surgical patches reviewers can ship without reading
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="How it works" />
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Supervised Fine-Tuning",
                desc: "500k (failure log, root cause, minimal fix) triples from 50k open source repos. Each fix is verified: CI went green and the diff passed a size budget. Greenlight learns failure taxonomy from real CI history.",
              },
              {
                step: "02",
                title: "RL with Verifiable Reward",
                desc: "Triple reward: CI green + diff size within budget + fix stable on rerun. All three signals are automatic and binary. Greenlight is punished for large diffs even if CI passes — teaching surgical precision.",
              },
              {
                step: "03",
                title: "DPO Alignment",
                desc: "Direct Preference Optimization on (stable fix, unstable fix) pairs. Greenlight learns to prefer targeted dep pins over broad upgrades, and test-scope isolation over test deletion.",
              },
            ].map(({ step, title, desc }) => {
              return (
                <div key={step} className="reveal-scale rounded-2xl border border-gray-100 bg-white p-8">
                  <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>{step}</div>
                  <h3 className="serif font-semibold text-lg mb-3 text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="Capabilities" />
        <div className="grid sm:grid-cols-2 gap-5">
          {[
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
              ),
              title: "Flaky test detection and elimination",
              desc: "Identifies timing-dependent, environment-dependent, and order-dependent test failures. Generates targeted fixes — sleep removal, proper mocking, or explicit ordering — not test deletion.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                </svg>
              ),
              title: "Dependency drift auto-remediation",
              desc: "Resolves breaking dep upgrades with minimum-change lockfile pins, understanding transitive dependency graphs. Won&apos;t upgrade when a pin solves it — teaches the model what minimal means.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
              ),
              title: "Environment-specific failure diagnosis",
              desc: "Distinguishes CI-only failures (Linux vs. macOS, different Node versions, missing env vars) from local failures. Generates targeted fixes per environment without touching unaffected paths.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>
                </svg>
              ),
              title: "Minimal-diff patch generation",
              desc: "Every fix is constrained to the smallest possible diff that resolves the failure. Greenlight CI won&apos;t refactor, reformat, or clean up — only what is necessary to go green.",
            },
          ].map(({ icon, title, desc }) => {
            return (
              <div
                key={title}
               
                className="reveal rounded-2xl border border-gray-100 p-7 flex gap-5 hover:border-gray-200 transition-colors"
              >
                <div
                  className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: `${ACCENT}10` }}
                >
                  {icon}
                </div>
                <div>
                  <h3 className="font-semibold text-sm text-gray-900 mb-1.5">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* The numbers */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="The numbers" />
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { stat: "500k", label: "Training pairs", sub: "From 50k open source repos" },
              { stat: "Qwen2.5-7B", label: "Base model", sub: "Coder-Instruct" },
              { stat: "3-part", label: "Reward signal", sub: "CI green + diff ≤ N + rerun stable" },
            ].map(({ stat, label, sub }) => {
              return (
                <div
                  key={label}
                 
                  className="reveal rounded-2xl border p-8"
                  style={{ borderColor: `${ACCENT}20` }}
                >
                  <div className="text-3xl font-bold tracking-tight mb-2" style={{ color: ACCENT }}>{stat}</div>
                  <div className="text-sm font-semibold text-gray-800 mb-1">{label}</div>
                  <div className="text-xs text-gray-400">{sub}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 border-t border-gray-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-400">
          <p>
            Part of the{" "}
            <a href={HUB_URL} className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio by{" "}
            <a
              href="https://github.com/calebnewtonusc"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-gray-600 transition-colors"
            >
              Caleb Newton &middot; calebnewtonusc
            </a>{" "}
            &middot; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
