import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const MODULE_DIR = path.join(process.cwd(), 'module3');
const UPLOAD_DIR = path.join(MODULE_DIR, 'uploads');

type PythonResult = {
  stdout: string;
  stderr: string;
};

async function ensureUploadDir() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
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

  throw lastError ?? new Error('Unable to find python interpreter');
}

function sanitizeFilename(name: string | null): string {
  if (!name) return 'upload.png';
  return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '');
}

async function saveUpload(file: Blob, fallbackName: string): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const filename = `${Date.now()}_${sanitizeFilename((file as any).name ?? fallbackName)}`;
  const filePath = path.join(UPLOAD_DIR, filename);
  await fs.writeFile(filePath, buffer);
  return filePath;
}

export async function POST(request: Request) {
  try {
    await ensureUploadDir();
    const formData = await request.formData();
    const operation = formData.get('operation');
    if (typeof operation !== 'string') {
      return NextResponse.json({ error: 'Missing operation' }, { status: 400 });
    }

    let args: string[] = ['-m', 'module3.cli', operation];
    if (operation === 'gradients' || operation === 'keypoints' || operation === 'boundary' || operation === 'aruco') {
      const blob = formData.get('image');
      if (!blob || !(blob instanceof Blob)) {
        return NextResponse.json({ error: 'Image upload required.' }, { status: 400 });
      }
      const saved = await saveUpload(blob, 'image.png');
      args = [...args, '--image', saved];
      if (operation === 'keypoints') {
        const mode = formData.get('mode');
        if (typeof mode === 'string') args.push('--mode', mode);
      }
      if (operation === 'aruco') {
        const dictionary = formData.get('dictionary');
        if (typeof dictionary === 'string') args.push('--dictionary', dictionary);
      }
    } else if (operation === 'compare') {
      const ref = formData.get('reference');
      const cand = formData.get('candidate');
      if (!(ref instanceof Blob) || !(cand instanceof Blob)) {
        return NextResponse.json({ error: 'Reference and candidate masks required.' }, { status: 400 });
      }
      const refPath = await saveUpload(ref, 'reference.png');
      const candPath = await saveUpload(cand, 'candidate.png');
      args = [...args, '--reference', refPath, '--candidate', candPath];
    } else {
      return NextResponse.json({ error: `Unsupported operation: ${operation}` }, { status: 400 });
    }

    const { stdout } = await spawnPython(args);
    const payload = JSON.parse(stdout);
    return NextResponse.json(payload);
  } catch (error: any) {
    console.error('[assignment3 API]', error);
    return NextResponse.json({ error: error?.message ?? 'Unexpected error' }, { status: 500 });
  }
}

