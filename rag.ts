import mammoth from 'mammoth';
import * as XLSX from 'xlsx';
import { db, newId } from '../../db/index.ts';
import { generateEmbedding, getGeminiClient } from './providers.ts';

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function lexicalScore(query: string, content: string): number {
  const terms = query
    .toLowerCase()
    .split(/[^a-z0-9\u0900-\u097f]+/)
    .filter((t) => t.length > 2);
  if (terms.length === 0) return 0;
  const lower = content.toLowerCase();
  let hits = 0;
  for (const term of terms) {
    if (lower.includes(term)) hits++;
  }
  return hits / terms.length;
}

export async function extractDocumentText(buffer: Buffer, filename: string, mimeType: string): Promise<string> {
  const lower = filename.toLowerCase();

  // 1. PDF extraction
  if (mimeType === 'application/pdf' || lower.endsWith('.pdf')) {
    try {
      const pdfParseModule = await import('pdf-parse');
      const pdfParse = (pdfParseModule as unknown as { default?: (b: Buffer) => Promise<{ text: string }> }).default ??
        (pdfParseModule as unknown as (b: Buffer) => Promise<{ text: string }>);
      const parsed = await pdfParse(buffer);
      if (parsed?.text?.trim()) {
        return parsed.text.trim();
      }
    } catch (err) {
      console.warn('pdf-parse failed, attempting Gemini multimodal PDF extraction:', err);
    }
    const ai = getGeminiClient();
    if (ai) {
      const resp = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            { inlineData: { mimeType: 'application/pdf', data: buffer.toString('base64') } },
            { text: 'Extract all readable text, headings, and tables from this PDF document verbatim, marking page breaks where visible.' },
          ],
        },
      });
      return resp.text?.trim() || 'Unable to extract text from PDF.';
    }
    throw new Error('PDF parsing failed and GEMINI_API_KEY is not configured.');
  }

  // 2. DOCX extraction
  if (
    mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    lower.endsWith('.docx')
  ) {
    const result = await mammoth.extractRawText({ buffer });
    return result.value.trim();
  }

  // 3. XLSX / CSV spreadsheet extraction
  if (
    lower.endsWith('.xlsx') ||
    lower.endsWith('.xls') ||
    lower.endsWith('.csv') ||
    mimeType.includes('spreadsheet') ||
    mimeType === 'text/csv'
  ) {
    const workbook = XLSX.read(buffer, { type: 'buffer' });
    const sheets: string[] = [];
    for (const sheetName of workbook.SheetNames) {
      const sheet = workbook.Sheets[sheetName];
      const csv = XLSX.utils.sheet_to_csv(sheet);
      sheets.push(`### Sheet: ${sheetName}\n${csv}`);
    }
    return sheets.join('\n\n').trim();
  }

  // 4. Image OCR + visual description via Gemini
  if (mimeType.startsWith('image/')) {
    const ai = getGeminiClient();
    if (!ai) {
      return '[Image uploaded - configure GEMINI_API_KEY to extract text and visual details]';
    }
    const resp = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: {
        parts: [
          { inlineData: { mimeType, data: buffer.toString('base64') } },
          { text: 'Extract all visible text, numbers, labels, and provide a structured factual description of this image.' },
        ],
      },
    });
    return resp.text?.trim() || '';
  }

  // 5. Plain text, Markdown, JSON, source code files
  return buffer.toString('utf-8').trim();
}

export function chunkText(text: string, chunkSize = 1100, overlap = 180): Array<{ content: string; sectionLabel: string }> {
  const cleaned = text.replace(/\r\n/g, '\n').trim();
  if (!cleaned) return [];

  const chunks: Array<{ content: string; sectionLabel: string }> = [];
  let start = 0;
  let idx = 1;

  while (start < cleaned.length) {
    const end = Math.min(start + chunkSize, cleaned.length);
    const slice = cleaned.slice(start, end).trim();
    if (slice) {
      const firstLine = slice.split('\n')[0]?.slice(0, 60) || `Section ${idx}`;
      chunks.push({
        content: slice,
        sectionLabel: `Section ${idx}: ${firstLine}`,
      });
      idx++;
    }
    if (end >= cleaned.length) break;
    start = end - overlap;
  }

  return chunks;
}

export async function indexDocumentChunks(params: {
  fileId: string;
  userId: string;
  text: string;
}): Promise<number> {
  const chunks = chunkText(params.text);
  const insertStmt = db.prepare(
    'INSERT INTO document_chunks (id, file_id, user_id, chunk_index, section_label, content, embedding_json) VALUES (?, ?, ?, ?, ?, ?, ?)'
  );

  // Generate embeddings for up to 24 chunks to stay responsive
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = i < 24 ? await generateEmbedding(chunk.content) : null;
    insertStmt.run(
      newId(),
      params.fileId,
      params.userId,
      i,
      chunk.sectionLabel,
      chunk.content,
      embedding ? JSON.stringify(embedding) : null
    );
  }

  return chunks.length;
}

export async function retrieveRelevantChunks(params: {
  userId: string;
  fileIds: string[];
  query: string;
  topK?: number;
}): Promise<Array<{ fileId: string; filename: string; sectionLabel: string; content: string; score: number }>> {
  if (params.fileIds.length === 0) return [];
  const placeholders = params.fileIds.map(() => '?').join(',');
  const rows = db
    .prepare(
      `SELECT dc.file_id, uf.filename, dc.section_label, dc.content, dc.embedding_json
       FROM document_chunks dc
       JOIN uploaded_files uf ON uf.id = dc.file_id
       WHERE dc.user_id = ? AND dc.file_id IN (${placeholders})`
    )
    .all(params.userId, ...params.fileIds) as Array<{
    file_id: string;
    filename: string;
    section_label: string;
    content: string;
    embedding_json: string | null;
  }>;

  if (rows.length === 0) return [];

  const queryEmbedding = await generateEmbedding(params.query);

  const scored = rows.map((row) => {
    let score = lexicalScore(params.query, row.content) * 0.4;
    if (queryEmbedding && row.embedding_json) {
      try {
        const vec = JSON.parse(row.embedding_json) as number[];
        score += cosineSimilarity(queryEmbedding, vec) * 0.6;
      } catch {
        // ignore bad embedding
      }
    } else {
      score = lexicalScore(params.query, row.content);
    }
    return {
      fileId: row.file_id,
      filename: row.filename,
      sectionLabel: row.section_label || 'Excerpt',
      content: row.content,
      score,
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, params.topK ?? 6);
}
