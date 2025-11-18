export default function AboutPage() {
  return (
    <main className="bg-slate-950 text-slate-50">
      <section className="mx-auto flex min-h-[60vh] w-full max-w-4xl flex-col gap-4 px-6 py-16">
        <h1 className="text-3xl font-semibold tracking-tight">About this toolkit</h1>
        <p className="text-sm text-slate-300 sm:text-base">
          This site wraps every CSC 8830 assignment into a single Next.js dashboard. Each module hooks into its matching Python experiment so you can
          run the workflow, capture screenshots, and export data for your report without leaving the browser.
        </p>
      </section>
    </main>
  );
}
