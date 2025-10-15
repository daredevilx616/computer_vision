import Navbar from "../components/Navbar";


export default function Home() {
  return (
    <main className="min-h-screen bg-gray-900 text-white">
      <Navbar />
      <div className="flex items-center justify-center h-[80vh]">
        <h1 className="text-5xl font-bold">Welcome to CV App 🚀</h1>
      </div>
    </main>
  );
}
