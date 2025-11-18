import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const MODULE_UPLOAD = path.join(process.cwd(), 'module4', 'uploads');

type PythonResult = { stdout: string; stderr: string };

async function ensureDirs() {
  await fs.mkdir(MODULE_UPLOAD, { recursive: true });
}

async function spawnPython(args: string[]): Promise<PythonResult> {
  const candidates = [
    process.env.PYTHON_BIN,
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : undefined,
    'python',
    'python3',
    'py',
  ].filter(Boolean) as string[];
  let lastError: unknown = null;
  for (const command of candidates) {
    try {
      return await new Promise<PythonResult>((resolve, reject) => {
        const child = spawn(command, args, { cwd: process.cwd() });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (chunk) => (stdout += chunk.toString()));
        child.stderr.on('data', (chunk) => (stderr += chunk.toString()));
        child.on('close', (code) => {
          if (code === 0) resolve({ stdout, stderr });
          else reject(new Error(stderr || `Python exited with code ${code}`));
        });
        child.on('error', reject);
      });
    } catch (error: any) {
      if (error?.code === 'ENOENT') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }
  throw lastError ?? new Error('Unable to locate python interpreter.');
}

function sanitize(name: string | null): string {
  if (!name) return 'image.png';
  return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '');
}

async function saveBlob(blob: Blob, fallback: string): Promise<string> {
  const buffer = Buffer.from(await blob.arrayBuffer());
  const filename = `${Date.now()}_${sanitize((blob as any).name ?? fallback)}`;
  const filePath = path.join(MODULE_UPLOAD, filename);
  await fs.writeFile(filePath, buffer);
  return filePath;
}

export async function POST(request: Request) {
  try {
    await ensureDirs();
    const formData = await request.formData();
    const images = formData.getAll('images').filter((value): value is File => value instanceof File);
    if (images.length < 2) {
      return NextResponse.json({ error: 'Upload at least two images for stitching.' }, { status: 400 });
    }
    const filePaths = await Promise.all(images.map((img, idx) => saveBlob(img, `frame_${idx}.png`)));
    const args = ['-m', 'module4.cli', 'stitch', '--images', ...filePaths];
    const { stdout } = await spawnPython(args);
    const payload = JSON.parse(stdout);
    return NextResponse.json(payload);
  } catch (error: any) {
    console.error('[assignment4/stitch]', error);
    return NextResponse.json({ error: error?.message ?? 'Unexpected error' }, { status: 500 });
  }
}
