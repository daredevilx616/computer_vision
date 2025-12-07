'use client';
import React, { useEffect, useRef, useState } from 'react';

export default function Page() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [photo, setPhoto] = useState<HTMLImageElement | null>(null);
  const [streamActive, setStreamActive] = useState(false);
  const [status, setStatus] = useState('idle');

  const [points, setPoints] = useState<{x:number;y:number}[]>([]);
  const [pixelDist, setPixelDist] = useState<number | null>(null);

  const [Zmeters, setZmeters] = useState('0.5');
  const [focalMM, setFocalMM] = useState('3.6');
  const [sensorWidthMM, setSensorWidthMM] = useState('3.2');
  const [resultMM, setResultMM] = useState<number | null>(null);

  const log = (...args:any[]) => console.log('[measure]', ...args);

  const startCamera = async () => {
    setStatus('requesting camera…');
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      setStream(s);
      setStatus('got stream, binding to video…');
      if (videoRef.current) {
        const v = videoRef.current;
        v.srcObject = s;
        // some browsers need the play call after metadata is ready
        v.onloadedmetadata = async () => {
          setStatus(`metadata: ${v.videoWidth}x${v.videoHeight}, calling play()…`);
          try {
            await v.play(); // may throw if autoplay policy blocks
            setStreamActive(true);
            setStatus('video playing ✅');
          } catch (e) {
            setStatus('autoplay blocked — press Play ▶ below');
            log('play() failed', e);
          }
        };
      }
    } catch (err) {
      setStatus('camera error — check permissions');
      alert('Unable to access camera. Check browser permission, close other apps using the camera, then retry.');
      log('getUserMedia error', err);
    }
  };

  const stopCamera = () => {
    stream?.getTracks().forEach(t => t.stop());
    setStream(null);
    setStreamActive(false);
    setStatus('stopped');
  };

  const manualPlay = async () => {
    if (!videoRef.current) return;
    try {
      await videoRef.current.play();
      setStreamActive(true);
      setStatus('video playing (manual) ✅');
    } catch (e) {
      setStatus('could not start video');
      log('manual play failed', e);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const v = videoRef.current, c = canvasRef.current;
    c.width = v.videoWidth || 1280;
    c.height = v.videoHeight || 720;
    const ctx = c.getContext('2d');
    if (ctx) ctx.drawImage(v, 0, 0);
    const url = c.toDataURL('image/png');
    const img = new Image();
    img.onload = () => setPhoto(img);
    img.src = url;
    setPoints([]); setPixelDist(null); setResultMM(null);
    // After capturing, stop the camera so the feed is frozen on the photo.
    stopCamera();
  };

  useEffect(() => {
    // draw photo & annotations
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0,0,canvasRef.current.width, canvasRef.current.height);
    if (photo) {
      canvasRef.current.width = photo.naturalWidth;
      canvasRef.current.height = photo.naturalHeight;
      ctx.drawImage(photo,0,0);
      ctx.strokeStyle = 'lime'; ctx.lineWidth = 3; ctx.font = '18px sans-serif';
      points.forEach((p,i)=>{ ctx.beginPath(); ctx.arc(p.x,p.y,6,0,Math.PI*2); ctx.stroke(); ctx.fillText(String(i+1), p.x+8, p.y-8); });
      if (points.length===2){ ctx.beginPath(); ctx.moveTo(points[0].x,points[0].y); ctx.lineTo(points[1].x,points[1].y); ctx.stroke(); }
    }
  }, [photo, points]);

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!photo || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width / rect.width;
    const sy = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;
    setPoints(prev=>{
      let next = [...prev, {x,y}];
      if (next.length>2) next=[{x,y}];
      if (next.length===2) {
        const d = Math.hypot(next[1].x-next[0].x, next[1].y-next[0].y);
        setPixelDist(d);
      } else setPixelDist(null);
      setResultMM(null);
      return next;
    });
  };

  const compute = () => {
    if (!photo || !pixelDist) return;
    const Zmm = parseFloat(Zmeters)*1000;
    const fmm = parseFloat(focalMM);
    const swmm = parseFloat(sensorWidthMM);
    const pmmPerPx = swmm / photo.naturalWidth;
    const Smm = (Zmm / fmm) * pmmPerPx * pixelDist;
    setResultMM(Smm);
  };

  // expose debug in console
  useEffect(() => {
    (window as any).__cam = { stream, video: videoRef.current };
  }, [stream]);

  return (
    <div className="min-h-screen p-6 text-white" style={{background:'#111'}}>
      <h1 className="text-2xl font-semibold mb-4">Real-World Measurement Demo (Live Camera)</h1>

      <div className="grid gap-6 lg:gap-10 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.85fr)]">
        <div>
          <div className="mb-3 text-sm opacity-80">Status: {status}</div>

          {!streamActive && (
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <button onClick={startCamera} className="border px-4 py-2 rounded">Start Camera</button>
              <button onClick={manualPlay} className="border px-4 py-2 rounded">Play (if blocked)</button>
              <button onClick={stopCamera} className="border px-4 py-2 rounded">Stop</button>
            </div>
          )}

          {/* Video preview */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            controls            // show play button just in case
            style={{ width:'640px', maxWidth:'95vw', borderRadius:12, background:'#000', display: stream ? 'block' : 'none' }}
          />

          {/* Capture & measure section */}
          {stream && (
            <div className="mt-3 flex items-center gap-3">
              <button onClick={capturePhoto} className="border px-4 py-2 rounded">Capture Photo</button>
              <button onClick={stopCamera} className="border px-4 py-2 rounded">Stop Camera</button>
            </div>
          )}

          <div className="w-full max-w-4xl mt-6">
            <div className="flex flex-wrap items-end gap-3 mb-3">
              <div className="flex flex-col">
                <label className="text-sm">Z (meters)</label>
                <input className="border border-gray-500 bg-white rounded px-2 py-1 text-black" value={Zmeters} onChange={e=>setZmeters(e.target.value)} />
              </div>
              <div className="flex flex-col">
                <label className="text-sm">Focal length f (mm)</label>
                <input className="border border-gray-500 bg-white rounded px-2 py-1 text-black" value={focalMM} onChange={e=>setFocalMM(e.target.value)} />
              </div>
              <div className="flex flex-col">
                <label className="text-sm">Sensor width (mm)</label>
                <input className="border border-gray-500 bg-white rounded px-2 py-1 text-black" value={sensorWidthMM} onChange={e=>setSensorWidthMM(e.target.value)} />
              </div>
              <button onClick={compute} className="border rounded px-3 py-2">Compute</button>
            </div>

            <canvas ref={canvasRef} onClick={onCanvasClick} style={{ width:'100%', borderRadius:10, background:'#000' }} />

            <div className="mt-2 text-sm">
              Pixel distance: {pixelDist ? pixelDist.toFixed(2) : '-'}
            </div>

            {resultMM !== null && (
              <div className="border rounded p-4 text-lg mt-3">
                <div className="font-semibold mb-1">Estimated real-world size</div>
                <div>{resultMM.toFixed(2)} mm</div>
                <div>{(resultMM/10).toFixed(2)} cm</div>
                <div>{(resultMM/25.4).toFixed(2)} in</div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-8 w-full max-w-4xl space-y-3 rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
          <p className="text-sm font-semibold text-slate-100">Recorded walkthrough</p>
          <div className="w-full overflow-hidden rounded border border-slate-800 bg-black aspect-video">
            <iframe
              className="w-full h-full"
              src="https://www.youtube.com/embed/Gz6gL7Fl0mE"
              title="Module 1 demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </div>
      </div>
    </div>
  );
}
