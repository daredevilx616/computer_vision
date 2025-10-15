import Navbar from "@/components/Navbar";

export default function About() {
  return (
    <main className="min-h-screen bg-gray-900 text-white">
      <Navbar />
      <div className="flex items-center justify-center h-[80vh]">
        <h1 className="text-4xl font-bold">About Page 📝</h1>
      </div>
    </main>
  );
}
