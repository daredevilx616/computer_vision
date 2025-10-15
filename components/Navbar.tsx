// components/Navbar.tsx
export default function Navbar() {
  return (
    <nav className="flex justify-between items-center p-4 bg-gray-800 text-white">
      <h1 className="text-xl font-bold">CV App</h1>
      <ul className="flex gap-6">
        <li><a href="/" className="hover:text-blue-400">Home</a></li>
        <li><a href="/about" className="hover:text-blue-400">About</a></li>
        <li><a href="/contact" className="hover:text-blue-400">Contact</a></li>
      </ul>
    </nav>
  );
}
